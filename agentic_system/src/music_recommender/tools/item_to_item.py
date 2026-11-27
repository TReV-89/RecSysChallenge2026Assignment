"""Item-to-item retrieval tool, scoped to one turn."""

from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from ._common import as_tool_content, summarize_candidates


def make_item_to_item_tool(retriever, request, collected: list):
    @tool
    def item_to_item_similarity(
        track_id: Annotated[str, Field(min_length=1)],
        vector_space: Literal["metadata", "cf"],
        top_k: Annotated[int, Field(ge=1, le=100)],
    ) -> str:
        """Find tracks similar to one known catalog track ("more like that track").

        track_id must be one you actually saw in this conversation (a prior
        tool result or the listener context below) - never a guess; an
        unknown ID raises an error. Use ``metadata`` for similar style, or
        ``cf`` for tracks tied to similar listener preferences.

        Args:
            track_id: Catalog ID of the example track.
            vector_space: ``metadata`` or ``cf``.
            top_k: Max tracks to return, 1-100.

        Returns:
            Up to top_k similar tracks (track_id, title, artists, tags),
            excluding the example; may be empty.
        """
        store = retriever.store
        if not store.state("ready"):
            raise RuntimeError("The catalog is not ready. Complete ingestion first.")
        if track_id not in store.records("tracks", [track_id]):
            raise ValueError("The example track ID is not in the catalog.")

        collection = "tracks" if vector_space == "metadata" else "tracks_cf"
        vector = store.vectors(collection, [track_id]).get(track_id)
        if vector is None:
            return as_tool_content([])

        matches = store.search(collection, vector, top_k + 1)
        catalog_tracks = store.records("tracks", matches)
        ids = [
            candidate_id
            for candidate_id in matches
            if candidate_id != track_id and candidate_id in catalog_tracks
        ][:top_k]
        result = retriever.item_candidates(request, ids, vector_space)
        collected.append(result)
        return as_tool_content(summarize_candidates(result))

    return item_to_item_similarity
