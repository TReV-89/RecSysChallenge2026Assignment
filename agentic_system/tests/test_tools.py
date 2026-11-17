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


def test_model_tool_schemas_hide_application_context_and_include_return_contracts():
    for tool, fields in zip(
        build_tools(StubRetriever()), [{"query"}, {"exclude_track_ids"}], strict=True
    ):
        function = convert_to_openai_tool(tool)["function"]
        assert set(function["parameters"]["properties"]) == fields
        assert "Returns:" in function["description"]
        assert "semantic_affinity" in function["description"]


def test_tools_return_typed_results_and_reject_unknown_ids():
    search, rank = build_tools(StubRetriever())
    request = RecommendationRequest(message="t1")
    result = search.invoke({"query": "calm", "request": request})
    assert isinstance(result, RetrievalResult)
    context = {"request": request, "retrieval": result}
    assert rank.invoke({"exclude_track_ids": [], **context})[0].score == 0.9
    assert rank.invoke({"exclude_track_ids": ["t1"], **context}) == []
    with pytest.raises(ValueError, match="outside the retrieved candidates"):
        rank.invoke({"exclude_track_ids": ["invented"], **context})
    with pytest.raises(ValidationError):
        search.invoke({"query": "", "request": request})


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

    tools = build_tools(StubRetriever())
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
