"""Shared helpers for request-scoped ReAct tools."""

import json

from ..schemas import RetrievalResult


def summarize_candidates(result: RetrievalResult) -> list[dict]:
    """Compact, catalog-grounded view of a retrieval result for the model."""
    return [
        {
            "track_id": candidate.track.track_id,
            "title": candidate.track.title,
            "artists": candidate.track.artists,
            "tags": candidate.track.tags[:12],
        }
        for candidate in result.candidates
    ]


def as_tool_content(value: list) -> str:
    """JSON-encode a tool's list result so it is always valid ToolMessage content.

    An empty Python list vacuously passes LangChain's "is this a list of
    content blocks" check (`all()` over zero items is `True`), so it is sent
    to the provider as an empty content array - which Groq (and other
    OpenAI-compatible APIs) reject outright. Returning a JSON string instead
    always yields valid, non-empty content, empty or not.
    """
    return json.dumps(value)
