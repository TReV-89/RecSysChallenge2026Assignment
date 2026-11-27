"""Candidate retrieval by source, followed by track-ID fusion and ranking."""

from .embeddings import cosine01
from .schemas import (
    Candidate,
    Neighbor,
    RecommendationRequest,
    RetrievalResult,
    SupportingUser,
    Track,
)
from .tools.calculate_user_similarity import find_cf_neighbors


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if item is not None and str(item)]


def track_from_row(row: dict) -> Track:
    titles = as_list(row.get("track_name") or row.get("track_name_primary"))
    artists = as_list(row.get("artist_name") or row.get("artist_name_primary"))
    return Track(
        track_id=row["track_id"],
        title=titles[0] if titles else "Untitled track",
        artists=artists,
        tags=list(dict.fromkeys(as_list(row.get("tag_list")))),
    )


class Retriever:
    def __init__(self, settings, store, embedder):
        self.settings, self.store, self.embedder = settings, store, embedder
        self._training_cf_vectors = None

    def _require_catalog(self) -> None:
        if not self.store.state("ready"):
            raise RuntimeError("The catalog is not ready. Complete ingestion first.")

    @staticmethod
    def _empty(query: str = "") -> RetrievalResult:
        return RetrievalResult(query=query, neighbors=[], candidates=[])

    def _query_cf_vector(self, request: RecommendationRequest):
        """Use a published test factor when present, otherwise a training factor."""
        if not request.user_id:
            return None
        query_record = (
            self.store.records("cf_query_vectors", [request.user_id]).get(request.user_id)
            if self.store.state("cf_query_vectors_ready")
            else None
        )
        if query_record is not None:
            return query_record["vector"]
        return self.store.vectors("users_cf", [request.user_id]).get(request.user_id)

    def _excluded_cf_tracks(self, request: RecommendationRequest) -> set[str]:
        excluded = set(request.excluded_track_ids)
        history_kind = "cf_history_" + self.settings.cf_history_mode
        if request.user_id and self.store.state(f"{history_kind}_ready"):
            own_history = self.store.records(history_kind, [request.user_id]).get(request.user_id)
            if own_history:
                excluded.update(own_history["track_ids"])
        return excluded

    def retrieve_text(self, request: RecommendationRequest, query: str) -> RetrievalResult:
        """Find tracks matching the current description in Qwen metadata space."""
        self._require_catalog()
        vector = self.embedder.query(query, "tracks")
        track_filter = (
            {"track_id": {"$nin": request.excluded_track_ids}}
            if request.excluded_track_ids
            else None
        )
        ids = self.store.search("tracks", vector, self.settings.candidate_count, track_filter)
        ids = [track_id for track_id in ids if track_id not in request.excluded_track_ids]
        rows = self.store.records("tracks", ids)
        vectors = self.store.vectors("tracks", ids)
        candidates = []
        for rank, track_id in enumerate(ids, 1):
            if track_id not in rows or track_id not in vectors:
                continue
            candidates.append(
                Candidate(
                    track=track_from_row(rows[track_id]),
                    query_score=cosine01(vector, vectors[track_id]),
                    neighbor_score=0.0,
                    supporting_users=[],
                    source_ranks={"text": rank},
                )
            )
        return RetrievalResult(query=query, neighbors=[], candidates=candidates)

    def retrieve_cf(self, request: RecommendationRequest) -> RetrievalResult:
        """Find tracks supported by eligible feedback from CF-BPR neighbors."""
        self._require_catalog()
        history_kind = "cf_history_" + self.settings.cf_history_mode
        if not request.user_id or not self.store.state(f"{history_kind}_ready"):
            return self._empty()
        user_vector = self._query_cf_vector(request)
        if user_vector is None:
            return self._empty()
        if self._training_cf_vectors is None:
            self._training_cf_vectors = self.store.all_vectors("users_cf")
        matches = find_cf_neighbors(request.user_id, user_vector, self._training_cf_vectors)
        histories = self.store.records(history_kind, [user_id for user_id, _ in matches])
        excluded = self._excluded_cf_tracks(request)
        support_scores: dict[str, float] = {}
        supporters: dict[str, list[SupportingUser]] = {}
        neighbors = []
        for user_id, score in matches:
            history = histories.get(user_id)
            if history is None:
                continue
            if len(neighbors) < self.settings.neighbor_count:
                neighbors.append(Neighbor(user_id=user_id, score=score, source="cf_bpr"))
            for track_id in history["track_ids"]:
                if track_id in excluded:
                    continue
                support_scores[track_id] = support_scores.get(track_id, 0.0) + score
                supporters.setdefault(track_id, []).append(
                    SupportingUser(
                        user_id=user_id,
                        similarity=score,
                        evidence_type="positive_conversation_feedback",
                    )
                )
        ordered = sorted(support_scores, key=lambda key: (-support_scores[key], key))
        available = self.store.existing_record_ids("tracks", ordered)
        ids = [track_id for track_id in ordered if track_id in available][
            : self.settings.candidate_count
        ]
        rows = self.store.records("tracks", ids)
        candidates = []
        for rank, track_id in enumerate(ids, 1):
            if track_id not in rows:
                continue
            evidence = sorted(
                supporters[track_id], key=lambda item: (-item.similarity, item.user_id)
            )[:3]
            candidates.append(
                Candidate(
                    track=track_from_row(rows[track_id]),
                    query_score=0.0,
                    neighbor_score=support_scores[track_id],
                    supporting_users=evidence,
                    source_ranks={"user_to_user": rank},
                )
            )
        return RetrievalResult(query="", neighbors=neighbors, candidates=candidates)

    def retrieve_user_to_item(self, request: RecommendationRequest) -> RetrievalResult:
        """Rank catalog tracks by inner product with the user's CF-BPR factor."""
        self._require_catalog()
        user_vector = self._query_cf_vector(request)
        if user_vector is None:
            return self._empty()
        collection = self.store.collections["tracks_cf"]
        count = collection.count()
        if not count:
            return self._empty()
        excluded = self._excluded_cf_tracks(request)
        fetch_count = min(count, max(1, self.settings.candidate_count * 2))
        while True:
            ids = self.store.search("tracks_cf", user_vector, fetch_count)
            rows = self.store.records("tracks", ids)
            eligible = [track_id for track_id in ids if track_id in rows and track_id not in excluded]
            if len(eligible) >= self.settings.candidate_count or fetch_count == count:
                break
            fetch_count = min(count, fetch_count * 2)
        candidates = [
            Candidate(
                track=track_from_row(rows[track_id]),
                query_score=0.0,
                neighbor_score=0.0,
                supporting_users=[],
                source_ranks={"user_to_item": rank},
            )
            for rank, track_id in enumerate(eligible[: self.settings.candidate_count], 1)
        ]
        return RetrievalResult(query="", neighbors=[], candidates=candidates)

    def item_candidates(
        self, request: RecommendationRequest, ids: list[str], vector_space: str
    ) -> RetrievalResult:
        """Attach catalog metadata to the item tool's ordered track IDs."""
        self._require_catalog()
        ids = [
            track_id
            for track_id in dict.fromkeys(ids)
            if track_id not in request.excluded_track_ids
        ]
        rows = self.store.records("tracks", ids)
        candidates = [
            Candidate(
                track=track_from_row(rows[track_id]),
                query_score=0.0,
                neighbor_score=0.0,
                supporting_users=[],
                source_ranks={f"item_to_item_{vector_space}": rank},
            )
            for rank, track_id in enumerate(ids, 1)
            if track_id in rows
        ]
        return RetrievalResult(query="", neighbors=[], candidates=candidates)

    def merge_results(
        self, request: RecommendationRequest, query: str, results: list[RetrievalResult]
    ) -> RetrievalResult:
        """Deduplicate tracks and sum reciprocal ranks across available sources."""
        merged: dict[str, Candidate] = {}
        neighbors: dict[str, Neighbor] = {}
        for result in results:
            for neighbor in result.neighbors:
                neighbors[neighbor.user_id] = neighbor
            for candidate in result.candidates:
                track_id = candidate.track.track_id
                if track_id in request.excluded_track_ids:
                    continue
                if track_id not in merged:
                    merged[track_id] = candidate.model_copy(deep=True)
                    continue
                current = merged[track_id]
                current.source_ranks.update(candidate.source_ranks)
                current.query_score = max(current.query_score, candidate.query_score)
                current.neighbor_score = max(current.neighbor_score, candidate.neighbor_score)
                existing_users = {user.user_id for user in current.supporting_users}
                current.supporting_users.extend(
                    user for user in candidate.supporting_users if user.user_id not in existing_users
                )
        candidates = list(merged.values())
        for candidate in candidates:
            candidate.supporting_users.sort(key=lambda user: (-user.similarity, user.user_id))
            candidate.supporting_users = candidate.supporting_users[:3]
            candidate.fusion_score = sum(
                1 / (60 + rank) for rank in candidate.source_ranks.values()
            )
        candidates.sort(key=lambda item: (-item.fusion_score, item.track.track_id))
        return RetrievalResult(
            query=query,
            neighbors=sorted(neighbors.values(), key=lambda n: (-n.score, n.user_id))[
                : self.settings.neighbor_count
            ],
            candidates=candidates[: self.settings.candidate_count],
        )

    def retrieve(self, request: RecommendationRequest, query: str) -> RetrievalResult:
        """Compatibility entry point for evaluation and non-graph callers."""
        return self.merge_results(
            request, query,
            [self.retrieve_text(request, query), self.retrieve_cf(request),
             self.retrieve_user_to_item(request)],
        )
