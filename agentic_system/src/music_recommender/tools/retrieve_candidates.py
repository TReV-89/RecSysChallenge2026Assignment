"""Catalog search tool, with the request supplied by the application."""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from ..schemas import RecommendationRequest, RetrievalResult


def create_retrieval_tool(retriever):
    """Bind catalog access without exposing the retriever to the language model."""

    @tool
    def retrieve_candidates(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
        request: Annotated[RecommendationRequest, InjectedToolArg],
    ) -> RetrievalResult:
        """Search the music catalog and find similar training listeners.

        Args:
            query: Music search incorporating the latest request, relevant prior
                preferences and dislikes. Use at most 2,000 characters.
            request: Application-provided listener context, exclusions and weights;
                the model does not supply or change this argument.

        Returns:
            RetrievalResult with query (str), neighbors (list of user_id,
            session_id, score, profile_score, goal_score, specificity_score), and
            candidates (list of track, query_score, neighbor_score,
            supporting_users). Each track has track_id, title, artists and tags.
            Supporting users have user_id, similarity, track_affinity and
            evidence_type='semantic_affinity'. Scores are relative similarities,
            not probabilities or recorded likes. Lists may be empty. Pass only
            returned candidate IDs to rank_tracks; never invent metadata.
        """
        return retriever.retrieve(request, query)

    return retrieve_candidates
