"""Check tool contracts in isolation, without a full ReAct loop."""

import json

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from music_recommender.ranking import rank_candidates
from music_recommender.schemas import (
    Candidate,
    RecommendationRequest,
    RetrievalResult,
    Track,
)
from music_recommender.tools import build_tools
from music_recommender.tools.calculate_user_similarity import find_cf_neighbors


class StubRetriever:
    def retrieve_text(self, request, query):
        return RetrievalResult(
            query=query,
            neighbors=[],
            candidates=[
                Candidate(
                    track=Track(track_id=request.message, title="Quiet", artists=["A"], tags=[]),
                    query_score=0.9,
                    neighbor_score=0,
                    supporting_users=[],
                    source_ranks={"text": 1},
                )
            ],
        )

    def retrieve_cf(self, request):
        return RetrievalResult(query="", neighbors=[], candidates=[])

    def retrieve_user_to_item(self, request):
        return RetrievalResult(query="", neighbors=[], candidates=[])

    def merge_results(self, request, query, results):
        merged = {}
        for result in results:
            for candidate in result.candidates:
                merged.setdefault(candidate.track.track_id, candidate)
        return RetrievalResult(query=query, neighbors=[], candidates=list(merged.values()))


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
    request = RecommendationRequest(message="t1")
    tools = build_tools(StubRetriever(), request, [], {})
    for tool, fields in zip(
        tools,
        [{"query"}, set(), set(), {"track_id", "vector_space", "top_k"}, {"exclude_track_ids"}],
        strict=True,
    ):
        function = convert_to_openai_tool(tool)["function"]
        assert set(function["parameters"]["properties"]) == fields
        assert "Returns:" in function["description"]


def test_tools_return_catalog_grounded_summaries_and_collect_results():
    request = RecommendationRequest(message="t1")
    collected = []
    result_box = {}
    search, user_to_user, user_to_item, _, rank = build_tools(
        StubRetriever(), request, collected, result_box
    )
    # Every tool returns a JSON string, never a raw (possibly empty) list: an empty
    # Python list passes LangChain's "is this a content block list" check vacuously
    # and gets sent to providers as invalid empty message content.
    summary = json.loads(search.invoke({"query": "calm"}))
    assert summary == [{"track_id": "t1", "title": "Quiet", "artists": ["A"], "tags": []}]
    assert json.loads(user_to_user.invoke({})) == []
    assert json.loads(user_to_item.invoke({})) == []
    assert len(collected) == 3  # text + empty cf + empty user-to-item, each appended

    ranked_summary = json.loads(rank.invoke({"exclude_track_ids": []}))
    assert ranked_summary[0]["track_id"] == "t1"
    assert result_box["ranked"][0].score == 0.9

    with pytest.raises(ValueError, match="outside the retrieved candidates"):
        rank_candidates(result_box["retrieval"], request, ["invented"])


def test_search_tool_rejects_empty_query():
    request = RecommendationRequest(message="calm")
    search = build_tools(StubRetriever(), request, [], {})[0]
    with pytest.raises(ValidationError):
        search.invoke({"query": ""})


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

    class FakeRetriever(StubRetriever):
        def __init__(self):
            self.store = FakeStore()

        def item_candidates(self, request, ids, vector_space):
            return RetrievalResult(
                query="",
                neighbors=[],
                candidates=[
                    Candidate(
                        track=Track(track_id=track_id, title=track_id, artists=[], tags=[]),
                        query_score=0.0,
                        neighbor_score=0.0,
                        supporting_users=[],
                        source_ranks={f"item_to_item_{vector_space}": rank},
                    )
                    for rank, track_id in enumerate(ids, 1)
                ],
            )

    request = RecommendationRequest(message="calm")
    item_tool = build_tools(FakeRetriever(), request, [], {})[3]
    result = json.loads(
        item_tool.invoke({"track_id": "example", "vector_space": "metadata", "top_k": 2})
    )
    assert result == [{"track_id": "similar", "title": "similar", "artists": [], "tags": []}]
    with pytest.raises(ValueError, match="not in the catalog"):
        item_tool.invoke({"track_id": "invented", "vector_space": "metadata", "top_k": 2})
