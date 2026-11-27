"""Deterministic ranking and diversity after candidate-source fusion."""

from .schemas import Candidate, RankedTrack, RecommendationRequest, RetrievalResult


def candidate_score(candidate: Candidate) -> float:
    """Use fused source ranks, falling back to a direct text score for older callers."""
    return candidate.fusion_score if candidate.fusion_score is not None else candidate.query_score


def rank_candidates(
    result: RetrievalResult, request: RecommendationRequest, excluded: list[str]
) -> list[RankedTrack]:
    known = {candidate.track.track_id for candidate in result.candidates}
    if not set(excluded).issubset(known):
        raise ValueError("Ranking agent tried to exclude IDs outside the retrieved candidates.")
    rejected = set(excluded) | set(request.excluded_track_ids)
    ranked = [
        RankedTrack(candidate=candidate, score=candidate_score(candidate))
        for candidate in result.candidates
        if candidate.track.track_id not in rejected
    ]
    ranked.sort(key=lambda item: (-item.score, item.candidate.track.track_id))
    selected, artist_counts = [], {}
    for item in ranked:
        artist = (item.candidate.track.artists or [item.candidate.track.track_id])[0].casefold()
        if artist_counts.get(artist, 0) >= 2:
            continue
        selected.append(item)
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        if len(selected) == request.limit:
            break
    return selected
