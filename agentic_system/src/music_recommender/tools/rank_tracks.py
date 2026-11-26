"""Deterministic ranking tool for already retrieved candidates."""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from ..schemas import RankedTrack, RecommendationRequest, RetrievalResult


@tool
def rank_tracks(
    exclude_track_ids: list[str],
    retrieval: Annotated[RetrievalResult, InjectedToolArg],
    request: Annotated[RecommendationRequest, InjectedToolArg],
) -> list[RankedTrack]:
    """Rank retrieved tracks using numerical scores and artist diversity.

    Call after the available retrieval sources have been merged. Exclude a track
    only when its catalog metadata clearly conflicts with an explicit preference.

    Args:
        exclude_track_ids: Retrieved candidate IDs to omit; use [] if none.
            Unknown IDs raise ValueError.
        retrieval: Application-provided merged candidate result.
        request: Application-provided listener weights, exclusions and limit.
            The model cannot supply or change retrieval or request.

    Returns:
        An ordered list of RankedTrack objects, each with score (float) and
        candidate (track, query_score, neighbor_score, source_ranks,
        supporting_users). Track contains track_id, title, artists and tags.
        When prepared CF data is loaded, rankings use reciprocal ranks from
        text and user-to-user retrieval. Supporting users then reflect positive
        synthetic conversation feedback, not a verified listen or like.
        Highest scores come first, ties use track_id, and at most two tracks per
        primary artist are selected, up to the requested limit. The list can be
        empty. Scores express relative fit, not probabilities or recorded likes.
    """
    from ..ranking import rank_candidates

    return rank_candidates(retrieval, request, exclude_track_ids)
