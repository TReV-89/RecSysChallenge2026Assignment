"""Text-to-item catalog search tool, scoped to one request's ReAct turn."""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from ._common import as_tool_content, summarize_candidates


def make_retrieve_candidates_tool(retriever, request, collected: list):
    @tool
    def retrieve_candidates(
        query: Annotated[str, Field(min_length=1, max_length=2000)],
    ) -> str:
        """Find catalog tracks from a text description (mood, genre, artist, activity).

        May be called again with different phrasing if results were a poor fit.

        Args:
            query: Search text (request + relevant preferences), max 2000 chars.

        Returns:
            Up to the candidate count, each with track_id, title, artists,
            tags; may be empty. These track_ids are catalog-grounded.
        """
        result = retriever.retrieve_text(request, query)
        collected.append(result)
        return as_tool_content(summarize_candidates(result))

    return retrieve_candidates
