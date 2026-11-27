import numpy as np
import pytest
from conftest import FakeEmbedder, ScriptedLLM, vector

from music_recommender.agent import build_agent, validate_explanations
from music_recommender.config import Settings
from music_recommender.embeddings import cosine01, validate_vector
from music_recommender.ingest import ingest, session_record
from music_recommender.ranking import rank_candidates
from music_recommender.retrieval import Retriever
from music_recommender.schemas import (
    Candidate,
    ExplanationDraft,
    RecommendationRequest,
    RetrievalResult,
    Track,
    TrackExplanation,
)
from music_recommender.store import CatalogStore


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


def test_agent_requires_an_llm(catalog):
    settings, store, *_ = catalog
    agent = build_agent(settings, Retriever(settings, store, FakeEmbedder()))
    with pytest.raises(RuntimeError, match="An LLM is required"):
        agent.invoke({"request": RecommendationRequest(message="ambient"), "notices": []})


def test_agent_skips_tools_for_a_greeting(catalog):
    settings, *_ = catalog

    class NoSearch:
        def retrieve_text(self, request, query):
            raise AssertionError("A greeting must not search the catalog")

    llm = ScriptedLLM(script=[{"text": "Hi! Tell me what music you're after."}])
    response = build_agent(settings, NoSearch(), llm).invoke(
        {"request": RecommendationRequest(message="Hello!"), "notices": []}
    )["response"]
    assert response.kind == "chat"
    assert response.message == "Hi! Tell me what music you're after."
    assert response.recommendations == []


def test_agent_runs_text_retrieval_then_ranks_and_explains(catalog):
    settings, store, *_ = catalog
    llm = ScriptedLLM(
        script=[
            {"tool": "retrieve_candidates", "args": {"query": "ambient"}},
            {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
            {"text": "Here is a calm pick."},
        ],
        explanation=ExplanationDraft(
            tracks=[
                TrackExplanation(track_id="t1", explanation="Its ambient tag fits.", cited_tags=["ambient"]),
            ]
        ),
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    response = build_agent(settings, retriever, llm).invoke(
        {"request": RecommendationRequest(message="ambient", limit=1), "notices": []}
    )["response"]
    assert llm.tool_calls_made == ["retrieve_candidates", "rank_tracks"]
    assert response.mode == "groq"
    assert response.notices == []
    assert [r.track.track_id for r in response.recommendations] == ["t1"]
    assert response.recommendations[0].explanation == "Its ambient tag fits."


def test_agent_can_select_cf_item_space_for_a_track_seen_in_context(catalog):
    settings, store, *_ = catalog
    llm = ScriptedLLM(
        script=[
            {
                "tool": "item_to_item_similarity",
                "args": {"track_id": "t1", "vector_space": "cf", "top_k": 5},
            },
            {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
            {"text": "Here is something with a similar collaborative profile."},
        ],
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    result = build_agent(settings, retriever, llm).invoke(
        {
            "request": RecommendationRequest(
                message="Find tracks with the same vibe", reference_track_id="t1"
            ),
            "notices": [],
        }
    )
    assert "item_to_item_similarity" in llm.tool_calls_made
    assert any(
        "item_to_item_cf" in candidate.source_ranks for candidate in result["retrieval"].candidates
    )


def test_item_to_item_rejects_an_id_not_seen_in_this_conversation(catalog):
    settings, store, *_ = catalog
    llm = ScriptedLLM(
        script=[
            {
                "tool": "item_to_item_similarity",
                "args": {"track_id": "invented-id", "vector_space": "metadata", "top_k": 5},
            },
            {"text": "I could not find that track; could you name it again?"},
        ],
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    response = build_agent(settings, retriever, llm).invoke(
        {"request": RecommendationRequest(message="more like that one"), "notices": []}
    )["response"]
    # The tool call raises ValueError inside the loop; the ReAct harness recovers by turning
    # it into a ToolMessage the model sees (no top-level failure), and no candidates were ranked.
    assert response.recommendations == []
    assert not any("Agent tool loop unavailable" in notice for notice in response.notices)
    assert response.message == "I could not find that track; could you name it again?"


def test_agent_resolves_a_reference_from_resent_client_history(catalog):
    settings, store, *_ = catalog
    from music_recommender.schemas import ChatTurn

    llm = ScriptedLLM(
        script=[
            {
                "tool": "item_to_item_similarity",
                "args": {"track_id": "t1", "vector_space": "metadata", "top_k": 5},
            },
            {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
            {"text": "Here is another ambient pick."},
        ],
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    request = RecommendationRequest(
        message="More like that one",
        history=[
            ChatTurn(role="user", content="Recommend something ambient"),
            ChatTurn(role="assistant", content="TrackID: t1 - Quiet by A"),
        ],
    )
    result = build_agent(settings, retriever, llm).invoke({"request": request, "notices": []})
    assert "item_to_item_similarity" in llm.tool_calls_made
    assert any(
        "item_to_item_metadata" in candidate.source_ranks
        for candidate in result["retrieval"].candidates
    )


def test_provider_failure_yields_empty_recommendations_with_a_notice(catalog):
    settings, store, *_ = catalog

    class ProviderError(RuntimeError):
        status_code = 413

    class UnavailableLLM:
        def bind_tools(self, *args, **kwargs):
            raise ProviderError("Simulated request too large")

        def with_structured_output(self, *args, **kwargs):
            raise ProviderError("Simulated request too large")

    retriever = Retriever(settings, store, FakeEmbedder())
    response = build_agent(settings, retriever, UnavailableLLM()).invoke(
        {"request": RecommendationRequest(message="ambient"), "notices": []}
    )["response"]
    assert response.recommendations == []
    assert any("Agent tool loop unavailable" in notice for notice in response.notices)
    assert any("HTTP 413" in notice for notice in response.notices)
    assert response.mode == "fallback"


def test_explanation_failure_falls_back_to_templates_but_keeps_recommendations(catalog):
    settings, store, *_ = catalog

    class NoExplanationLLM(ScriptedLLM):
        def with_structured_output(self, schema, **kwargs):
            raise RuntimeError("Simulated judge outage")

    llm = NoExplanationLLM(
        script=[
            {"tool": "retrieve_candidates", "args": {"query": "ambient"}},
            {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
            {"text": "Here is a calm pick."},
        ]
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    response = build_agent(settings, retriever, llm).invoke(
        {"request": RecommendationRequest(message="ambient", limit=1), "notices": []}
    )["response"]
    assert response.mode == "fallback"
    assert response.recommendations
    assert any("Agent explanation unavailable" in notice for notice in response.notices)


def test_checkpointed_thread_does_not_resend_history(catalog):
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    settings, store, *_ = catalog
    retriever = Retriever(settings, store, FakeEmbedder())
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "conversation-123"}}

    first_llm = ScriptedLLM(script=[{"text": "ok"}])
    build_agent(settings, retriever, first_llm, checkpointer=checkpointer).invoke(
        {"request": RecommendationRequest(message="ambient"), "notices": []}, config=config
    )
    second_llm = ScriptedLLM(script=[{"text": "ok again"}])
    build_agent(settings, retriever, second_llm, checkpointer=checkpointer).invoke(
        {"request": RecommendationRequest(message="anything else?"), "notices": []}, config=config
    )

    probe = create_agent(second_llm, tools=[], checkpointer=checkpointer)
    snapshot = probe.get_state(config)
    # Each turn adds exactly one HumanMessage and one final AIMessage; two turns means four
    # persisted messages, never the client's full history re-added on top of the checkpoint.
    assert len(snapshot.values["messages"]) == 4


def test_long_local_query_fits_search_contract(catalog):
    settings, store, *_ = catalog
    request = RecommendationRequest(message="x" * 6000, profile="p" * 6000, goal="g" * 3000)
    llm = ScriptedLLM(
        script=[
            {"tool": "retrieve_candidates", "args": {"query": "x" * 2000}},
            {"tool": "rank_tracks", "args": {"exclude_track_ids": []}},
            {"text": "ok"},
        ]
    )
    retriever = Retriever(settings, store, FakeEmbedder())
    response = build_agent(settings, retriever, llm).invoke({"request": request, "notices": []})["response"]
    assert response.kind in {"chat", "recommendations"}


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
