"""Documented LangChain tools used by the recommendation agent."""

from .item_to_item import item_to_item_similarity
from .rank_tracks import rank_tracks
from .retrieve_candidates import retrieve_candidates
from .retrieve_user_to_user_candidates import retrieve_user_to_user_candidates


def build_tools():
    """Return the agent's tools; injected catalog context is passed at execution."""
    return [retrieve_candidates, retrieve_user_to_user_candidates, item_to_item_similarity, rank_tracks]
