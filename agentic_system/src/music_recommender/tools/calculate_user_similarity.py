"""CF-BPR user-neighbor calculation used by collaborative retrieval."""

from collections.abc import Mapping, Sequence

import numpy as np


def find_cf_neighbors(
    query_user_id: str,
    query_vector: Sequence[float] | None,
    training_vectors: Mapping[str, Sequence[float]],
    *,
    min_similarity: float = 0.0,
    max_neighbors: int | None = None,
    center_vector: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Rank training users by cosine or population-centered cosine.

    Vectors may be original CF-BPR factors or profile-enhanced factors. When a
    center is supplied, subtract that same training-population mean from the
    query and every neighbor before cosine. The production retriever supplies
    enhanced user factors and this center by default. No rating means are involved.
    Every supplied training user is compared before filtering or truncation.

    Args:
        query_user_id: User whose neighbors are requested.
        query_vector: This user's 128-number user factor, or None if unavailable.
        training_vectors: Training user ID to user factor mapping; do not
            include validation or test-only users in this pool.
        min_similarity: Keep scores strictly above this similarity threshold.
            The default of zero keeps positively aligned users; choose a
            stronger threshold using held-out validation, not by assumption.
        max_neighbors: Optional result cap applied after scoring all training
            users. None returns every qualifying user, with no top-30 limit.
        center_vector: Mean of the supplied enhanced training factors for
            population-centered cosine; None uses ordinary cosine.

    Returns:
        Pairs of (training_user_id, similarity), in descending
        similarity order with user ID breaking ties. Returns [] when the query
        vector is unavailable or no training user passes the threshold.

    Raises:
        ValueError: A supplied vector or center is invalid, or a threshold or
            result cap is invalid.
    """
    if not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between -1 and 1.")
    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError("max_neighbors must be positive when supplied.")
    if query_vector is None:
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    if query.shape != (128,) or not np.isfinite(query).all() or not np.linalg.norm(query):
        raise ValueError("The query CF-BPR vector must be finite, nonzero and 128-dimensional.")
    if center_vector is not None:
        center = np.asarray(center_vector, dtype=np.float32)
        if center.shape != (128,) or not np.isfinite(center).all():
            raise ValueError("The training-user center must be finite and 128-dimensional.")
        query = query - center
        if not np.linalg.norm(query):
            return []
    else:
        center = None
    query = query / np.linalg.norm(query)

    neighbors = []
    for user_id, vector in training_vectors.items():
        if user_id == query_user_id:
            continue
        candidate = np.asarray(vector, dtype=np.float32)
        if (
            candidate.shape != (128,)
            or not np.isfinite(candidate).all()
            or not np.linalg.norm(candidate)
        ):
            raise ValueError(f"Invalid CF-BPR vector for training user {user_id}.")
        if center is not None:
            candidate = candidate - center
            if not np.linalg.norm(candidate):
                continue
        score = float(np.clip(np.dot(query, candidate / np.linalg.norm(candidate)), -1, 1))
        if score > min_similarity:
            neighbors.append((user_id, score))

    neighbors.sort(key=lambda item: (-item[1], item[0]))
    return neighbors[:max_neighbors]
