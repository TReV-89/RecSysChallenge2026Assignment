"""Request-scoped LangChain tool factories used by the ReAct recommendation agent."""

from .item_to_item import make_item_to_item_tool
from .rank_tracks import make_rank_tracks_tool
from .retrieve_candidates import make_retrieve_candidates_tool
from .retrieve_user_to_item_candidates import make_retrieve_user_to_item_tool
from .retrieve_user_to_user_candidates import make_retrieve_user_to_user_tool


def build_tools(retriever, request, collected: list, result_box: dict):
    """Build one request's tool list; collected/result_box are private to this turn."""
    return [
        make_retrieve_candidates_tool(retriever, request, collected),
        make_retrieve_user_to_user_tool(retriever, request, collected),
        make_retrieve_user_to_item_tool(retriever, request, collected),
        make_item_to_item_tool(retriever, request, collected),
        make_rank_tracks_tool(retriever, request, collected, result_box),
    ]
