from langchain_core.tools import tool

from .embeddings import cosine01
from .schemas import (
    Candidate,
    Neighbor,
    RankedTrack,
    RecommendationRequest,
    RetrievalResult,
    SimilarityWeights,
    SupportingUser,
    Track,
)


def weighted_similarity(profile, goal, specificity, weights: SimilarityWeights) -> float:
    pairs = [(profile, weights.profile), (goal, weights.goal), (specificity, weights.specificity)]
    denominator = sum(weight for score, weight in pairs if score is not None)
    if denominator == 0:
        return 0.0
    return sum(score * weight for score, weight in pairs if score is not None) / denominator


@tool
def calculate_user_similarity(
    profile_score: float | None,
    goal_score: float | None,
    specificity_score: float | None,
    weights: SimilarityWeights,
) -> float:
    """Combine profile, goal and specificity similarities; renormalize missing components."""
    for value in (profile_score, goal_score, specificity_score):
        if value is not None and not 0 <= value <= 1:
            raise ValueError("Similarity components must be between zero and one.")
    return weighted_similarity(profile_score, goal_score, specificity_score, weights)


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item is not None and str(item)]


def track_from_row(row: dict) -> Track:
    titles = as_list(row.get("track_name"))
    return Track(
        track_id=row["track_id"],
        title=titles[0] if titles else "Untitled track",
        artists=as_list(row.get("artist_name")),
        tags=list(dict.fromkeys(as_list(row.get("tag_list")))),
    )


class Retriever:
    def __init__(self, settings, store, embedder):
        self.settings, self.store, self.embedder = settings, store, embedder

    def similar_users(self, request: RecommendationRequest, query: str) -> list[Neighbor]:
        profile_vector = (
            self.embedder.query(request.profile, "profiles") if request.profile.strip() else None
        )
        goal_text = f"{request.goal}\nCurrent request: {query}".strip()
        goal_vector = self.embedder.query(goal_text, "goals")
        where = {"user_id": {"$ne": request.user_id}} if request.user_id else None
        pool = set(self.store.search("goals", goal_vector, self.settings.neighbor_pool, where))
        if profile_vector is not None:
            pool.update(
                self.store.search("profiles", profile_vector, self.settings.neighbor_pool, where)
            )
        if request.session_id:
            pool.discard(request.session_id)
        ids = sorted(pool)
        sessions = self.store.records("sessions", ids)
        profiles = self.store.vectors("profiles", ids)
        goals = self.store.vectors("goals", ids)
        best_per_user = {}
        for session_id in ids:
            row = sessions.get(session_id)
            if row is None or row["user_id"] == request.user_id or row["split"] != "train":
                continue
            p = (
                cosine01(profile_vector, profiles[session_id])
                if profile_vector is not None and session_id in profiles
                else None
            )
            g = cosine01(goal_vector, goals[session_id]) if session_id in goals else None
            s = (
                float(request.specificity == row["specificity"])
                if request.specificity and row["specificity"]
                else None
            )
            score = calculate_user_similarity.invoke(
                {
                    "profile_score": p,
                    "goal_score": g,
                    "specificity_score": s,
                    "weights": request.weights,
                }
            )
            neighbor = Neighbor(
                user_id=row["user_id"],
                session_id=session_id,
                score=score,
                profile_score=p,
                goal_score=g,
                specificity_score=s,
            )
            previous = best_per_user.get(neighbor.user_id)
            if previous is None or neighbor.score > previous.score:
                best_per_user[neighbor.user_id] = neighbor
        return sorted(best_per_user.values(), key=lambda n: (-n.score, n.user_id))[
            : self.settings.neighbor_count
        ]

    def retrieve(self, request: RecommendationRequest, query: str) -> RetrievalResult:
        if not self.store.state("ready"):
            raise RuntimeError("The catalog is not ready. Complete ingestion first.")
        query_vector = self.embedder.query(query, "tracks")
        neighbors = self.similar_users(request, query)
        profile_vectors = self.store.vectors("profiles", [n.session_id for n in neighbors])
        # Obtain direct matches plus candidates suggested by similar listeners' profiles.
        track_filter = (
            {"track_id": {"$nin": request.excluded_track_ids}}
            if request.excluded_track_ids
            else None
        )
        candidates = set(
            self.store.search("tracks", query_vector, self.settings.candidate_count, track_filter)
        )
        for neighbor in neighbors:
            if neighbor.session_id in profile_vectors:
                candidates.update(
                    self.store.search(
                        "tracks", profile_vectors[neighbor.session_id], 5, track_filter
                    )
                )
        candidates.difference_update(request.excluded_track_ids)
        ids = sorted(candidates)
        track_rows = self.store.records("tracks", ids)
        track_vectors = self.store.vectors("tracks", ids)
        results = []
        for track_id in ids:
            if track_id not in track_rows or track_id not in track_vectors:
                continue
            evidence = []
            for neighbor in neighbors:
                if neighbor.session_id in profile_vectors and neighbor.score > 0:
                    evidence.append(
                        SupportingUser(
                            user_id=neighbor.user_id,
                            similarity=neighbor.score,
                            track_affinity=cosine01(
                                profile_vectors[neighbor.session_id], track_vectors[track_id]
                            ),
                        )
                    )
            denominator = sum(e.similarity for e in evidence)
            affinity = (
                sum(e.similarity * e.track_affinity for e in evidence) / denominator
                if denominator
                else 0
            )
            evidence.sort(key=lambda e: (-e.similarity * e.track_affinity, e.user_id))
            results.append(
                Candidate(
                    track=track_from_row(track_rows[track_id]),
                    query_score=cosine01(query_vector, track_vectors[track_id]),
                    neighbor_score=affinity,
                    supporting_users=evidence[:3],
                )
            )
        # Bound the LLM context; preserve a mix by using the eventual numerical score.
        results.sort(key=lambda c: (-candidate_score(c, request.query_weight), c.track.track_id))
        return RetrievalResult(
            query=query, neighbors=neighbors, candidates=results[: self.settings.candidate_count]
        )


def candidate_score(candidate: Candidate, query_weight: float) -> float:
    if not candidate.supporting_users:
        return candidate.query_score
    return query_weight * candidate.query_score + (1 - query_weight) * candidate.neighbor_score


def rank_candidates(
    result: RetrievalResult, request: RecommendationRequest, excluded: list[str]
) -> list[RankedTrack]:
    known = {c.track.track_id for c in result.candidates}
    if not set(excluded).issubset(known):
        raise ValueError("Ranking agent tried to exclude IDs outside the retrieved candidates.")
    rejected = set(excluded) | set(request.excluded_track_ids)
    ranked = [
        RankedTrack(candidate=c, score=candidate_score(c, request.query_weight))
        for c in result.candidates
        if c.track.track_id not in rejected
    ]
    ranked.sort(key=lambda r: (-r.score, r.candidate.track.track_id))
    # Up to two songs per primary artist; stable ties make the numeric ranking repeatable.
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
