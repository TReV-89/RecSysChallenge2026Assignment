"""Exercise the prepared-data boundary and a complete CF-backed retrieval turn."""

import pandas as pd
from conftest import ScriptedLLM

from music_recommender.agent import build_agent
from music_recommender.cf_data import load_history, load_test_query_vectors
from music_recommender.config import Settings
from music_recommender.retrieval import Retriever
from music_recommender.schemas import RecommendationRequest
from music_recommender.store import CatalogStore


def vector(size, axis=0):
    values = [0.0] * size
    values[axis] = 1.0
    return values


class Embedder:
    def query(self, text, purpose="tracks"):
        return vector(1024)


def test_greeting_bypasses_retrieval():
    class NoSearch:
        def retrieve_text(self, request, query):
            raise AssertionError("A greeting must not search the catalog")

    llm = ScriptedLLM(script=[{"text": "Hi! Tell me what music you're after."}])
    response = build_agent(Settings(), NoSearch(), llm).invoke(
        {"request": RecommendationRequest(message="Hello!"), "notices": []}
    )["response"]
    assert response.kind == "chat"
    assert response.message
    assert response.recommendations == []


def test_conversational_question_bypasses_retrieval_with_provider():
    class NoSearch:
        def retrieve_text(self, request, query):
            raise AssertionError("A discussion turn must not search the catalog")

    llm = ScriptedLLM(script=[{"text": "We can talk through your music preferences first."}])
    response = build_agent(Settings(), NoSearch(), llm).invoke(
        {"request": RecommendationRequest(message="I am curious about your approach"), "notices": []}
    )["response"]
    assert response.kind == "chat"
    assert "preferences" in response.message
    assert response.mode == "groq"


def test_official_history_and_test_vector_feed_cf_candidates(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "catalog", candidate_count=5)
    store = CatalogStore(settings)
    store.upsert(
        "tracks",
        ["text-track", "neighbor-track"],
        [vector(1024), vector(1024, 1)],
        [{"track_id": "text-track"}, {"track_id": "neighbor-track"}],
    )
    store.save_records(
        "tracks",
        [
            {"track_id": "text-track", "track_name": ["Text"], "artist_name": ["A"]},
            {"track_id": "neighbor-track", "track_name": ["Neighbor"], "artist_name": ["B"]},
        ],
        "track_id",
    )
    store.upsert("users_cf", ["train-a"], [vector(128)], [{"user_id": "train-a"}])
    store.upsert(
        "tracks_cf",
        ["text-track", "neighbor-track", "unindexed-track"],
        [vector(128), [2.0] + [0.0] * 127, [3.0] + [0.0] * 127],
        [{"track_id": "text-track"}, {"track_id": "neighbor-track"},
         {"track_id": "unindexed-track"}],
    )
    store.set_state("ready", True)
    pd.DataFrame(
        [
            {"user_id": "train-a", "session_id": "s1", "track_id": "neighbor-track",
             "source_split": "train", "is_positive": True, "is_labeled": True},
            {"user_id": "train-a", "session_id": "s1", "track_id": "aaa-unindexed",
             "source_split": "train", "is_positive": True, "is_labeled": True},
            {"user_id": "train-a", "session_id": "s2", "track_id": "text-track",
             "source_split": "train", "is_positive": False, "is_labeled": True},
            {"user_id": "test-q", "session_id": "s3", "track_id": "text-track",
             "source_split": "test", "is_positive": True, "is_labeled": True},
        ]
    ).to_parquet(tmp_path / "conversation_interactions.parquet")
    report = load_history(store, tmp_path, "official_test")
    assert report == {"rows": 2, "user_track_pairs": 2, "users": 1}
    assert store.records("cf_history_official_test", ["test-q"]) == {}
    counts = load_test_query_vectors(
        store,
        loader=lambda key, unused: (
            [{"user_id": "test-q", "cf-bpr": vector(128)}]
            if key == "test_warm_user_vectors"
            else [{"user_id": "no-vector", "cf-bpr": []}]
        ),
    )
    assert counts == {"test_warm": 1, "test_cold": 0, "missing": 1}

    retriever = Retriever(settings, store, Embedder())
    result = retriever.retrieve(RecommendationRequest(message="calm music", user_id="test-q"), "calm")
    by_id = {candidate.track.track_id: candidate for candidate in result.candidates}
    assert "neighbor-track" in by_id
    assert by_id["neighbor-track"].source_ranks["user_to_user"] == 1
    assert by_id["neighbor-track"].source_ranks["user_to_item"] == 1
    assert by_id["text-track"].source_ranks["user_to_item"] == 2
    assert by_id["neighbor-track"].supporting_users[0].evidence_type == "positive_conversation_feedback"
    assert result.neighbors[0].user_id == "train-a"
    assert result.neighbors[0].source == "cf_bpr"
    limited = Retriever(settings.model_copy(update={"candidate_count": 1}), store, Embedder())
    assert [candidate.track.track_id for candidate in limited.retrieve_cf(
        RecommendationRequest(message="calm music", user_id="test-q")
    ).candidates] == ["neighbor-track"]
    assert [candidate.track.track_id for candidate in limited.retrieve_user_to_item(
        RecommendationRequest(message="calm music", user_id="test-q")
    ).candidates] == ["neighbor-track"]
    assert [candidate.track.track_id for candidate in limited.retrieve_user_to_item(
        RecommendationRequest(message="calm music", user_id="test-q",
                              excluded_track_ids=["neighbor-track"])
    ).candidates] == ["text-track"]

    two_source_script = [
        {"tool": "retrieve_candidates", "args": {"query": "calm music"}},
        {"tool": "retrieve_user_to_user_candidates", "args": {}},
        {"tool": "retrieve_user_to_item_candidates", "args": {}},
        {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
        {"text": "Here is a calm pick from similar listeners."},
    ]

    response = build_agent(settings, retriever, ScriptedLLM(script=two_source_script)).invoke(
        {"request": RecommendationRequest(message="calm music", user_id="test-q"), "notices": []}
    )["response"]
    assert response.kind == "recommendations"
    assert any(item.track.track_id == "neighbor-track" for item in response.recommendations)

    from fastapi.testclient import TestClient

    from main import app

    monkeypatch.setattr(
        app.state,
        "agent",
        build_agent(settings, retriever, ScriptedLLM(script=two_source_script)),
        raising=False,
    )
    http = TestClient(app).post(
        "/chat", json={"message": "calm music", "user_id": "test-q"}
    )
    assert http.status_code == 200
    assert any(row["track"]["track_id"] == "neighbor-track" for row in http.json()["recommendations"])

    four_source_script = [
        {"tool": "retrieve_candidates", "args": {"query": "more like this track"}},
        {"tool": "retrieve_user_to_user_candidates", "args": {}},
        {"tool": "retrieve_user_to_item_candidates", "args": {}},
        {
            "tool": "item_to_item_similarity",
            "args": {"track_id": "text-track", "vector_space": "metadata", "top_k": 20},
        },
        {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
        {"text": "Here is something similar from every available source."},
    ]
    three_source = build_agent(settings, retriever, ScriptedLLM(script=four_source_script)).invoke(
        {
            "request": RecommendationRequest(
                message="More like this track",
                user_id="test-q",
                reference_track_id="text-track",
            ),
            "notices": [],
        }
    )
    neighbor_candidate = next(
        candidate for candidate in three_source["retrieval"].candidates
        if candidate.track.track_id == "neighbor-track"
    )
    assert "user_to_user" in neighbor_candidate.source_ranks
    assert "user_to_item" in neighbor_candidate.source_ranks
    assert "item_to_item_metadata" in neighbor_candidate.source_ranks
    assert "text" in neighbor_candidate.source_ranks

    cold = retriever.retrieve(RecommendationRequest(message="calm music", user_id="no-vector"), "calm")
    assert cold.neighbors == []
    assert all("user_to_user" not in item.source_ranks for item in cold.candidates)
    assert all("user_to_item" not in item.source_ranks for item in cold.candidates)
