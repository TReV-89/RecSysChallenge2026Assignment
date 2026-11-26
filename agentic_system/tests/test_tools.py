"""Check tool contracts and the main agent without a catalog or provider request."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from music_recommender.agent import build_agent, default_query, tool_arguments
from music_recommender.config import Settings
from music_recommender.schemas import (
    Candidate,
    RecommendationRequest,
    RetrievalResult,
    SearchArguments,
    Track,
)
from music_recommender.tools import build_tools
from music_recommender.tools.calculate_user_similarity import find_cf_neighbors


class StubRetriever:
    def retrieve(self, request, query):
        return RetrievalResult(
            query=query,
            neighbors=[],
            candidates=[
                Candidate(
                    track=Track(track_id=request.message, title="Quiet", artists=["A"], tags=[]),
                    query_score=0.9,
                    neighbor_score=0,
                    supporting_users=[],
                )
            ],
        )


def test_cf_neighbors_scores_full_training_pool_before_optional_cutoff():
    query = [1.0] + [0.0] * 127
    training = {
        "self": query,
        "close": [0.8, 0.6] + [0.0] * 126,
        "closest": [1.0] + [0.0] * 127,
        "opposite": [-1.0] + [0.0] * 127,
    }
    ranked = find_cf_neighbors("self", query, training)
    assert [user_id for user_id, _ in ranked] == ["closest", "close"]
    assert [score for _, score in ranked] == pytest.approx([1.0, 0.8])
    assert find_cf_neighbors("self", query, training, max_neighbors=1) == ranked[:1]
    assert find_cf_neighbors("self", query, training, min_similarity=0.9) == ranked[:1]
    assert find_cf_neighbors("self", None, training) == []


def test_model_tool_schemas_hide_application_context_and_include_return_contracts():
    for tool, fields in zip(
        build_tools(),
        [{"query"}, set(), {"track_id", "vector_space", "top_k"}, {"exclude_track_ids"}],
        strict=True,
    ):
        function = convert_to_openai_tool(tool)["function"]
        assert set(function["parameters"]["properties"]) == fields
        assert "Returns:" in function["description"]


def test_tools_return_typed_results_and_reject_unknown_ids():
    search, user_search, _, rank = build_tools()
    request = RecommendationRequest(message="t1")
    result = search.invoke({"query": "calm", "request": request, "retriever": StubRetriever()})
    assert isinstance(result, RetrievalResult)
    class CFStub:
        def retrieve_cf(self, supplied_request):
            assert supplied_request is request
            return RetrievalResult(query="", neighbors=[], candidates=[])

    assert user_search.invoke({"request": request, "retriever": CFStub()}).candidates == []
    context = {"request": request, "retrieval": result}
    assert rank.invoke({"exclude_track_ids": [], **context})[0].score == 0.9
    assert rank.invoke({"exclude_track_ids": ["t1"], **context}) == []
    with pytest.raises(ValueError, match="outside the retrieved candidates"):
        rank.invoke({"exclude_track_ids": ["invented"], **context})
    with pytest.raises(ValidationError):
        search.invoke({"query": "", "request": request, "retriever": StubRetriever()})


def test_item_to_item_searches_selected_space_and_excludes_example():
    class FakeStore:
        def state(self, key):
            return True

        def records(self, kind, ids):
            return {track_id: {} for track_id in ids if track_id in {"example", "similar"}}

        def vectors(self, kind, ids):
            return {"example": [1.0]} if kind == "tracks" else {}

        def search(self, kind, vector, limit):
            assert (kind, vector, limit) == ("tracks", [1.0], 3)
            return ["example", "similar", "unknown"]

    retriever = StubRetriever()
    retriever.store = FakeStore()
    item_tool = build_tools()[2]
    assert item_tool.invoke(
        {"track_id": "example", "vector_space": "metadata", "top_k": 2, "store": retriever.store}
    ) == ["similar"]
    assert item_tool.invoke(
        {"track_id": "example", "vector_space": "cf", "top_k": 2, "store": retriever.store}
    ) == []
    with pytest.raises(ValueError, match="not in the catalog"):
        item_tool.invoke(
            {"track_id": "invented", "vector_space": "metadata", "top_k": 2, "store": retriever.store}
        )


def test_main_agent_reuse_keeps_request_context_isolated():
    agent = build_agent(Settings(), StubRetriever())
    for track_id in ("first", "second"):
        response = agent.invoke({"request": RecommendationRequest(message=track_id)})["response"]
        assert response.mode == "local"
        assert [item.track.track_id for item in response.recommendations] == [track_id]


def test_long_local_query_fits_search_contract():
    request = RecommendationRequest(message="x" * 6000, profile="p" * 6000, goal="g" * 3000)
    query = default_query(request)
    assert SearchArguments(query=query).query.endswith("x" * 2000)
    response = build_agent(Settings(), StubRetriever()).invoke({"request": request})["response"]
    assert response.mode == "local"


@pytest.mark.parametrize(
    "calls",
    [
        [],
        [{"name": "rank_tracks", "args": {"exclude_track_ids": []}, "id": "wrong"}],
        [{"name": "retrieve_candidates", "args": {"query": "calm", "request": {}}, "id": "injected"}],
        [
            {"name": "retrieve_candidates", "args": {"query": "calm"}, "id": "one"},
            {"name": "retrieve_candidates", "args": {"query": "calm"}, "id": "two"},
        ],
    ],
)
def test_main_agent_rejects_unexpected_or_context_overriding_calls(calls):
    class InvalidLLM:
        def bind_tools(self, tools, tool_choice):
            return RunnableLambda(lambda messages: AIMessage(content="", tool_calls=calls))

    tools = build_tools()
    with pytest.raises(ValueError):
        tool_arguments(InvalidLLM(), tools, tools[0], SearchArguments, "prompt", "calm")


def test_catalog_failures_are_not_hidden_as_provider_fallbacks():
    class BrokenRetriever:
        def retrieve(self, request, query):
            raise RuntimeError("Catalog unavailable")

    with pytest.raises(RuntimeError, match="Catalog unavailable"):
        build_agent(Settings(), BrokenRetriever()).invoke(
            {"request": RecommendationRequest(message="calm")}
        )
