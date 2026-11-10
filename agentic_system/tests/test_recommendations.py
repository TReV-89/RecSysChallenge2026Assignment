import numpy as np
import pytest
from pydantic import ValidationError

from music_recommender.config import Settings
from music_recommender.embeddings import cosine01, validate_vector
from music_recommender.graph import build_graph, validate_explanations
from music_recommender.ingest import ingest, session_record
from music_recommender.retrieval import Retriever, rank_candidates, weighted_similarity
from music_recommender.schemas import (
    Candidate,
    ExplanationDraft,
    RecommendationRequest,
    RetrievalResult,
    SimilarityWeights,
    Track,
    TrackExplanation,
)
from music_recommender.store import CatalogStore


def vector(axis=0, size=1024):
    result = np.zeros(size, dtype=np.float32)
    result[axis] = 1
    return result.tolist()


class FakeEmbedder:
    def documents(self, texts):
        return [vector(1 if "rock" in text.lower() else 0) for text in texts]

    def query(self, text, purpose="tracks"):
        return vector(1 if "rock" in text.lower() else 0)


@pytest.fixture
def catalog(tmp_path):
    settings = Settings(data_dir=tmp_path / "catalog", cache_dir=tmp_path / "cache")
    rows = {
        "sessions": [
            {
                "session_id": "s1",
                "user_id": "u1",
                "user_profile": {"preferred_musical_culture": "ambient"},
                "conversation_goal": {"listener_goal": "relax", "specificity": "LL"},
            },
            {
                "session_id": "s2",
                "user_id": "u1",
                "user_profile": {"preferred_musical_culture": "ambient"},
                "conversation_goal": {"listener_goal": "rest", "specificity": "LL"},
            },
            {
                "session_id": "s3",
                "user_id": "u2",
                "user_profile": {"preferred_musical_culture": "rock"},
                "conversation_goal": {"listener_goal": "rock", "specificity": "HH"},
            },
        ],
        "tracks": [
            {
                "track_id": "t1",
                "track_name": ["Quiet"],
                "artist_name": ["A"],
                "tag_list": ["ambient"],
            },
            {"track_id": "t2", "track_name": ["Loud"], "artist_name": ["B"], "tag_list": ["rock"]},
            {"track_id": "t3", "track_name": ["Still"], "artist_name": ["C"], "tag_list": ["calm"]},
        ],
        "users": [{"user_id": "u1"}, {"user_id": "u2"}],
        "track_vectors": [
            {
                "track_id": "t1",
                "metadata-qwen3_embedding_0.6b": vector(),
                "cf-bpr": vector(size=128),
            },
            {"track_id": "t2", "metadata-qwen3_embedding_0.6b": vector(1), "cf-bpr": []},
            {
                "track_id": "t3",
                "metadata-qwen3_embedding_0.6b": vector(),
                "cf-bpr": vector(size=128),
            },
            {"track_id": "empty", "metadata-qwen3_embedding_0.6b": [], "cf-bpr": []},
        ],
        "user_vectors": [{"user_id": "u1", "cf-bpr": vector(size=128)}],
    }
    loader = lambda key, cache_dir: rows[key]
    report = ingest(settings, loader=loader, embedder=FakeEmbedder())
    return settings, CatalogStore(settings), report, loader


def test_similarity_weights_and_missing_components():
    weights = SimilarityWeights()
    assert weighted_similarity(0.8, 0.4, 1, weights) == pytest.approx(0.69)
    assert weighted_similarity(None, 0.4, None, weights) == pytest.approx(0.4)
    assert weighted_similarity(None, None, None, weights) == 0
    with pytest.raises(ValidationError):
        SimilarityWeights(profile=0, goal=0, specificity=0)


def test_embedding_spaces_and_missing_values():
    assert validate_vector([], 1024) is None
    with pytest.raises(ValueError):
        validate_vector(vector(size=128), 1024)
    with pytest.raises(ValueError):
        validate_vector([float("nan")], 1)
    with pytest.raises(ValueError):
        cosine01(vector(), vector(size=128))


def test_ingestion_is_idempotent_and_keeps_spaces_separate(catalog):
    settings, store, report, loader = catalog
    assert report["missing_track_text"] == 1
    assert store.counts() == {"tracks": 3, "profiles": 3, "goals": 3, "tracks_cf": 2, "users_cf": 1}
    assert np.allclose(store.vectors("tracks", ["t1"])["t1"], vector())
    ingest(settings, loader=loader, embedder=FakeEmbedder())
    assert store.collections["tracks"].count() == 3
    with pytest.raises(ValueError, match="different data directories"):
        ingest(settings, sample=True, loader=loader, embedder=FakeEmbedder())


def test_session_extraction_does_not_index_conversation_answers():
    record = session_record(
        {
            "session_id": "s",
            "user_id": "u",
            "user_profile": {},
            "conversation_goal": {"specificity": "HH"},
            "conversations": [{"thought": "hidden", "content": "future answer"}],
            "goal_progress_assessments": ["future reward"],
        }
    )
    assert "conversations" not in record
    assert "goal_progress_assessments" not in record


def test_retrieval_deduplicates_users_and_excludes_self(catalog):
    settings, store, *_ = catalog
    retriever = Retriever(settings, store, FakeEmbedder())
    request = RecommendationRequest(message="ambient", profile="ambient", specificity="LL")
    result = retriever.retrieve(request, "ambient")
    assert [n.user_id for n in result.neighbors] == ["u1", "u2"]
    assert result.candidates[0].track.track_id == "t1"
    request.user_id = "u1"
    request.excluded_track_ids = ["t1"]
    result = retriever.retrieve(request, "ambient")
    assert all(n.user_id != "u1" for n in result.neighbors)
    assert "t1" not in [c.track.track_id for c in result.candidates]


def test_specificity_is_exact_match_and_unknown_is_omitted(catalog):
    settings, store, *_ = catalog
    retriever = Retriever(settings, store, FakeEmbedder())
    request = RecommendationRequest(message="ambient", specificity="HL")
    assert all(n.specificity_score == 0 for n in retriever.similar_users(request, "ambient"))
    request.specificity = None
    assert all(n.specificity_score is None for n in retriever.similar_users(request, "ambient"))


def test_ranker_and_explainer_reject_invented_ids_and_tags():
    result = RetrievalResult(
        query="calm",
        neighbors=[],
        candidates=[
            Candidate(
                track=Track(track_id="t", title="Quiet", artists=["A"], tags=["ambient"]),
                query_score=0.9,
                neighbor_score=0,
                supporting_users=[],
            )
        ],
    )
    request = RecommendationRequest(message="calm")
    with pytest.raises(ValueError):
        rank_candidates(result, request, ["invented"])
    ranked = rank_candidates(result, request, [])
    assert ranked[0].score == 0.9  # no artificial penalty for absent neighbors
    for track_id, tags in [("invented", []), ("t", ["jazz"])]:
        with pytest.raises(ValueError):
            validate_explanations(
                ExplanationDraft(
                    tracks=[
                        TrackExplanation(
                            track_id=track_id,
                            explanation="Fits",
                            cited_tags=tags,
                        )
                    ]
                ),
                ranked,
            )


def test_local_graph_runs_end_to_end(catalog):
    settings, store, *_ = catalog
    graph = build_graph(settings, Retriever(settings, store, FakeEmbedder()))
    response = graph.invoke({"request": RecommendationRequest(message="ambient"), "notices": []})[
        "response"
    ]
    assert response.mode == "local"
    assert response.recommendations[0].track.track_id == "t1"
    assert all(
        u.evidence_type == "semantic_affinity"
        for r in response.recommendations
        for u in r.supporting_users
    )


class UnavailableLLM:
    def bind_tools(self, *args, **kwargs):
        raise RuntimeError("Simulated rate limit")

    def with_structured_output(self, *args, **kwargs):
        raise RuntimeError("Simulated rate limit")


def test_provider_failure_returns_labeled_local_fallback(catalog):
    settings, store, *_ = catalog
    graph = build_graph(settings, Retriever(settings, store, FakeEmbedder()), UnavailableLLM())
    response = graph.invoke({"request": RecommendationRequest(message="ambient"), "notices": []})[
        "response"
    ]
    assert response.mode == "fallback"
    assert len(response.notices) == 3
    assert response.recommendations


class ScriptedLLM:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, tool_choice):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        assert len(tools) == 1
        self.calls.append(tool_choice)
        args = (
            {"query": "ambient"}
            if tool_choice == "retrieve_candidates"
            else {"exclude_track_ids": ["t2"]}
        )
        return RunnableLambda(
            lambda messages: AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_choice,
                        "args": args,
                        "id": "call1",
                        "type": "tool_call",
                    }
                ],
            )
        )

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        self.calls.append("explain")
        assert kwargs == {"method": "json_schema", "strict": True}
        return RunnableLambda(
            lambda messages: ExplanationDraft(
                tracks=[
                    TrackExplanation(
                        track_id="t1",
                        explanation="Its ambient tag suits this request.",
                        cited_tags=["ambient"],
                    ),
                    TrackExplanation(
                        track_id="t3",
                        explanation="Its calm tag suits this request.",
                        cited_tags=["calm"],
                    ),
                ]
            )
        )


def test_three_agent_tool_and_structured_output_path(catalog):
    settings, store, *_ = catalog
    llm = ScriptedLLM()
    graph = build_graph(settings, Retriever(settings, store, FakeEmbedder()), llm)
    response = graph.invoke({"request": RecommendationRequest(message="ambient"), "notices": []})[
        "response"
    ]
    assert llm.calls == ["retrieve_candidates", "rank_tracks", "explain"]
    assert response.mode == "groq"
    assert [r.track.track_id for r in response.recommendations] == ["t1", "t3"]
    assert response.notices == []


def test_ingestion_rejects_missing_metadata(tmp_path):
    settings = Settings(data_dir=tmp_path)
    rows = {
        "sessions": [
            {"session_id": "s", "user_id": "u", "user_profile": {}, "conversation_goal": {}}
        ],
        "track_vectors": [
            {"track_id": "orphan", "metadata-qwen3_embedding_0.6b": vector(), "cf-bpr": []}
        ],
        "tracks": [],
    }
    with pytest.raises(ValueError, match="Missing metadata"):
        ingest(settings, loader=lambda key, cache: rows[key], embedder=FakeEmbedder())
    assert CatalogStore(settings).state("ready") is False


def test_index_identity_mismatch_is_rejected(catalog):
    settings, *_ = catalog
    changed = settings.model_copy(update={"track_embedding_column": "lyrics-qwen3_embedding_0.6b"})
    with pytest.raises(ValueError, match="different embedding configuration"):
        CatalogStore(changed)
