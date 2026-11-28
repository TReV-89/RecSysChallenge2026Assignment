"""Apply the offline musical-culture correction to published CF-BPR user factors."""

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

STOPWORDS = {"culture", "music", "musical"}
PROFILE_FIELD = re.compile(
    r"(?:^|;)\s*preferred[_ ]musical[_ ]culture\s*:\s*([^;]+)", re.IGNORECASE
)


def culture_tokens(profile: str) -> set[str]:
    """Read only the culture field from the request's existing profile text."""
    match = PROFILE_FIELD.search(profile)
    if match is None:
        return set()
    return set(re.findall(r"[a-z0-9]+", match.group(1).casefold())) - STOPWORDS


class CultureAdapter:
    """Frozen culture-token weights and precomputed enhanced training users."""

    def __init__(self, weights_path: Path, users_path: Path):
        with np.load(weights_path, allow_pickle=False) as saved:
            tokens = [str(token) for token in saved["tokens"]]
            self.weights = np.asarray(saved["correction_weights"], dtype=np.float32)
            self.alphas = {
                "user_to_user": float(saved["user_to_user_alpha"]),
                "user_to_item": float(saved["user_to_item_alpha"]),
            }
            self.max_fraction = float(saved["maximum_correction_fraction"])
        if len(tokens) != len(set(tokens)) or self.weights.shape != (len(tokens), 128):
            raise ValueError("Invalid culture adapter vocabulary or weight shape")
        if not np.isfinite(self.weights).all() or not 0 <= self.max_fraction <= 1:
            raise ValueError("Invalid culture adapter weights or correction limit")
        if any(not np.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in self.alphas.values()):
            raise ValueError("Invalid culture adapter correction strength")
        self.vocabulary = {token: index for index, token in enumerate(tokens)}

        with np.load(users_path, allow_pickle=False) as saved:
            ids = [str(user_id) for user_id in saved["user_ids"]]
            self.training_vectors = np.asarray(saved["user_to_user_vectors"], dtype=np.float32)
            self.base_user_factors_sha256 = str(saved["base_user_factors_sha256"])
        if len(ids) != len(set(ids)) or self.training_vectors.shape != (len(ids), 128):
            raise ValueError("Invalid enhanced training-user vector archive")
        if not np.isfinite(self.training_vectors).all():
            raise ValueError("Enhanced training-user vectors must be finite")
        self.training_ids = ids
        self.training_center = self.training_vectors.mean(axis=0)

    def training_pool(
        self, original: Mapping[str, Sequence[float]]
    ) -> dict[str, np.ndarray]:
        """Reject an adapter built against a different published user catalog."""
        if set(original) != set(self.training_ids):
            raise ValueError("CF adapter training users do not match the catalog")
        matrix = np.asarray([original[user] for user in self.training_ids], dtype="<f4")
        actual_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
        if actual_hash != self.base_user_factors_sha256:
            raise ValueError("CF adapter was trained with different published user factors")
        return dict(zip(self.training_ids, self.training_vectors, strict=True))

    def enhance(self, vector, profile: str, route: str) -> list[float]:
        """Add a capped correction; an absent culture leaves the factor intact."""
        base = np.asarray(vector, dtype=np.float32)
        if base.shape != (128,) or not np.isfinite(base).all():
            raise ValueError("CF-BPR query factor must be finite and 128-dimensional")
        indexes = self._indexes(profile)
        if not indexes:
            return base.tolist()
        delta = self.weights[indexes].sum(axis=0) / np.sqrt(len(indexes))
        norm = float(np.linalg.norm(delta))
        limit = self.max_fraction * float(np.linalg.norm(base))
        if norm > limit and norm > 0:
            delta *= limit / norm
        return (base + self.alphas[route] * delta).tolist()

    def has_known_culture(self, profile: str) -> bool:
        """Whether this request can receive a learned culture correction."""
        return bool(self._indexes(profile))

    def _indexes(self, profile: str) -> list[int]:
        return [self.vocabulary[token] for token in culture_tokens(profile)
                if token in self.vocabulary]

