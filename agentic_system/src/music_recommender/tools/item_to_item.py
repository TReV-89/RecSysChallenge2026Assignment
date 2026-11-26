"""Item-to-item retrieval from one catalog embedding space."""

from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field


@tool
def item_to_item_similarity(
    track_id: Annotated[str, Field(min_length=1)],
    vector_space: Literal["metadata", "cf"],
    top_k: Annotated[int, Field(ge=1, le=100)],
    store: Annotated[object, InjectedToolArg],
) -> list[str]:
    """Find tracks similar to a known track the listener mentioned or selected.

    Call this for requests such as "more like that track" after identifying its
    catalog track ID from the conversation. Do not invent a track ID. Choose
    ``metadata`` when the listener wants a similar style or description, or
    ``cf`` when they want tracks associated with similar listener preferences.

    Args:
        track_id: Catalog ID of the example track.
        vector_space: Type of similarity to use: ``metadata`` or ``cf``.
        top_k: Maximum number of similar tracks to return, from 1 to 100.

    Returns:
        An ordered list of up to ``top_k`` similar catalog track IDs. The
        example track is excluded. An empty list means no similar tracks were
        available from this search; use another retrieval route if needed.
    """
    if not store.state("ready"):
        raise RuntimeError("The catalog is not ready. Complete ingestion first.")
    if track_id not in store.records("tracks", [track_id]):
        raise ValueError("The example track ID is not in the catalog.")

    collection = "tracks" if vector_space == "metadata" else "tracks_cf"
    vector = store.vectors(collection, [track_id]).get(track_id)
    if vector is None:
        return []

    matches = store.search(collection, vector, top_k + 1)
    catalog_tracks = store.records("tracks", matches)
    return [
        candidate_id
        for candidate_id in matches
        if candidate_id != track_id and candidate_id in catalog_tracks
    ][:top_k]
