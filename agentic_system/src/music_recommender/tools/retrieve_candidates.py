"""Catalog search tool, with the request supplied by the application."""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from ..schemas import RecommendationRequest, RetrievalResult


@tool
def retrieve_candidates(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    request: Annotated[RecommendationRequest, InjectedToolArg],
    retriever: Annotated[object, InjectedToolArg],
) -> RetrievalResult:
    """Find catalog tracks matching the listener's current music description.

    Args:
        query: Music search incorporating the latest request, relevant prior
            preferences and dislikes. Use at most 2,000 characters.
        request: Application-provided listener context and exclusions.
        retriever: Application-provided catalog retriever. The model cannot
            set or change either application argument.

    Returns:
        RetrievalResult with query and catalog-backed candidates ranked by
        metadata similarity. Each candidate includes its track ID, title,
        artists, tags and text source rank. This tool does not calculate
        user similarity. It may return an empty candidate list.
    """
    if hasattr(retriever, "retrieve_text"):
        return retriever.retrieve_text(request, query)
    return retriever.retrieve(request, query)
