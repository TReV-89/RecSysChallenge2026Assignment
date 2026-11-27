"""Deterministic ranking tool over everything retrieved so far this turn."""

from langchain_core.tools import tool

from ._common import as_tool_content


def make_rank_tracks_tool(retriever, request, collected: list, result_box: dict):
    @tool
    def rank_tracks(exclude_track_ids: list[str]) -> str:
        """Finalize the ranked track list from everything retrieved so far.

        Call exactly once, after retrieving and before your final answer.
        Merges/dedupes every retrieval tool's results this turn, scores them
        deterministically, and caps two tracks per artist. You do not set
        scores; only exclude candidates whose metadata clearly conflicts with
        an explicit listener preference.

        Args:
            exclude_track_ids: Retrieved IDs to omit; [] if none. An ID
                outside the retrieved candidates raises an error.

        Returns:
            Final ranked list (track_id, title, artists, tags, score); may
            be empty. Use these exact values; never invent a score.
        """
        from ..ranking import rank_candidates

        merged = retriever.merge_results(request, request.message, collected)
        ranked = rank_candidates(merged, request, exclude_track_ids)
        result_box["ranked"] = ranked
        result_box["retrieval"] = merged
        return as_tool_content(
            [
                {
                    "track_id": item.candidate.track.track_id,
                    "title": item.candidate.track.title,
                    "artists": item.candidate.track.artists,
                    "tags": item.candidate.track.tags[:12],
                    "score": round(item.score, 4),
                }
                for item in ranked
            ]
        )

    return rank_tracks
