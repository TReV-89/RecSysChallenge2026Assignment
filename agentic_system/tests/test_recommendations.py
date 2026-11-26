import numpy as np
import pytest

from music_recommender.agent import build_agent, known_reference, validate_explanations
from music_recommender.config import Settings
from music_recommender.embeddings import cosine01, validate_vector
from music_recommender.ingest import ingest, session_record
from music_recommender.ranking import rank_candidates
from music_recommender.retrieval import Retriever
from music_recommender.schemas import (
    Candidate,
    ChatTurn,
    ExplanationDraft,
    RecommendationRequest,
    RetrievalResult,
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


def test_text_retrieval_excludes_requested_tracks_without_cf_history(catalog):
    settings, store, *_ = catalog
    retriever = Retriever(settings, store, FakeEmbedder())
    request = RecommendationRequest(message="ambient", profile="ambient", specificity="LL")
    result = retriever.retrieve(request, "ambient")
    assert result.neighbors == []  # No CF history has been imported into this fixture.
    assert result.candidates[0].track.track_id == "t1"
    request.user_id = "u1"
    request.excluded_track_ids = ["t1"]
    result = retriever.retrieve(request, "ambient")
    assert all(n.user_id != "u1" for n in result.neighbors)
    assert "t1" not in [c.track.track_id for c in result.candidates]


def test_specificity_does_not_change_text_only_retrieval(catalog):
    settings, store, *_ = catalog
    retriever = Retriever(settings, store, FakeEmbedder())
    request = RecommendationRequest(message="ambient", specificity="HL")
    first = retriever.retrieve(request, "ambient")
    request.specificity = None
    second = retriever.retrieve(request, "ambient")
    assert [c.track.track_id for c in first.candidates] == [
        c.track.track_id for c in second.candidates
    ]


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
    graph = build_agent(settings, Retriever(settings, store, FakeEmbedder()))
    response = graph.invoke({"request": RecommendationRequest(message="ambient"), "notices": []})[
        "response"
    ]
    assert response.mode == "local"
    assert response.recommendations[0].track.track_id == "t1"
    assert response.neighbors == []


def test_agent_uses_item_to_item_for_verified_reference(catalog):
    settings, store, *_ = catalog
    graph = build_agent(settings, Retriever(settings, store, FakeEmbedder()))
    result = graph.invoke(
        {
            "request": RecommendationRequest(
                message="More like this track, please",
                reference_track_id="t1",
            )
        }
    )
    assert result["response"].kind == "recommendations"
    by_id = {candidate.track.track_id: candidate for candidate in result["retrieval"].candidates}
    assert "item_to_item_metadata" in by_id["t3"].source_ranks
    assert "t1" not in {
        candidate.track.track_id
        for candidate in result["retrieval"].candidates
        if "item_to_item_metadata" in candidate.source_ranks
    }


def test_unresolved_anaphoric_reference_asks_for_track():
    class NoSearch:
        def retrieve(self, request, query):
            raise AssertionError("Ambiguous reference triggered retrieval")

    response = build_agent(Settings(), NoSearch()).invoke(
        {"request": RecommendationRequest(message="More like that one")}
    )["response"]
    assert response.kind == "chat"
    assert "Which track" in response.message


def test_reference_resolves_from_validated_assistant_history():
    track_id = "12345678-1234-1234-1234-123456789abc"

    class Store:
        def records(self, kind, ids):
            return {track_id: {"track_id": track_id}} if track_id in ids else {}

    state = {
        "request": RecommendationRequest(
            message="More like that one",
            history=[ChatTurn(role="assistant", content=f"TrackID: {track_id}")],
        )
    }
    assert known_reference(state, Store()) == track_id


def test_single_prior_recommendation_can_be_item_reference(catalog):
    from langgraph.checkpoint.memory import InMemorySaver

    settings, store, *_ = catalog
    graph = build_agent(
        settings, Retriever(settings, store, FakeEmbedder()), checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": "item-followup"}}
    first = graph.invoke(
        {"request": RecommendationRequest(message="ambient", limit=1)}, config=config
    )
    reference = first["response"].recommendations[0].track.track_id
    second = graph.invoke(
        {"request": RecommendationRequest(message="More like that one")}, config=config
    )
    assert second["reference_track_id"] == reference
    assert any(
        "item_to_item_metadata" in candidate.source_ranks
        for candidate in second["retrieval"].candidates
    )


def test_agent_can_select_cf_item_space_for_known_track(catalog):
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    class ToolModel:
        def __init__(self):
            self.calls = []

        def bind_tools(self, tools, tool_choice):
            self.calls.append(tool_choice)
            choices = {
                "RouteDecision": {"route": "recommendations", "use_item_to_item": True},
                "retrieve_candidates": {"query": "calm"},
                "item_to_item_similarity": {
                    "track_id": "t1", "vector_space": "cf", "top_k": 5
                },
                "rank_tracks": {"exclude_track_ids": []},
            }
            return RunnableLambda(
                lambda messages: AIMessage(
                    content="",
                    tool_calls=[{"name": tool_choice, "args": choices[tool_choice], "id": "call"}],
                )
            )

        def with_structured_output(self, *args, **kwargs):
            raise RuntimeError("Use local explanations in this test")

    settings, store, *_ = catalog
    model = ToolModel()
    result = build_agent(settings, Retriever(settings, store, FakeEmbedder()), model).invoke(
        {"request": RecommendationRequest(message="Find tracks with the same vibe", reference_track_id="t1")}
    )
    assert "item_to_item_similarity" in model.calls
    assert any(
        "item_to_item_cf" in candidate.source_ranks
        for candidate in result["retrieval"].candidates
    )


def test_graph_saves_state_to_configured_checkpoint(catalog):
    from langgraph.checkpoint.memory import InMemorySaver

    settings, store, *_ = catalog
    graph = build_agent(
        settings,
        Retriever(settings, store, FakeEmbedder()),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "conversation-123"}}
    request = RecommendationRequest(message="ambient")

    graph.invoke({"request": request, "notices": []}, config=config)

    snapshot = graph.get_state(config)
    assert snapshot.values["request"] == request
    assert snapshot.values["response"]


class UnavailableLLM:
    def bind_tools(self, *args, **kwargs):
        raise RuntimeError("Simulated rate limit")

    def with_structured_output(self, *args, **kwargs):
        raise RuntimeError("Simulated rate limit")


def test_provider_failure_returns_labeled_local_fallback(catalog):
    settings, store, *_ = catalog
    graph = build_agent(settings, Retriever(settings, store, FakeEmbedder()), UnavailableLLM())
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

        assert {tool.name for tool in tools} == {
            "retrieve_candidates", "retrieve_user_to_user_candidates",
            "item_to_item_similarity", "rank_tracks"
        }
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


def test_main_agent_tool_and_structured_output_path(catalog):
    settings, store, *_ = catalog
    llm = ScriptedLLM()
    graph = build_agent(settings, Retriever(settings, store, FakeEmbedder()), llm)
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
