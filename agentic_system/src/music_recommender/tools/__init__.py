"""Documented LangChain tools used by the recommendation agent."""

from .rank_tracks import rank_tracks
from .retrieve_candidates import create_retrieval_tool


def build_tools(retriever):
    """Return the main agent's tools, bound to this catalog retriever."""
    return [create_retrieval_tool(retriever), rank_tracks]
