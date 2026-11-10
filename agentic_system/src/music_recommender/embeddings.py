from collections import OrderedDict
from collections.abc import Sequence
from threading import RLock

import numpy as np

from .config import Settings


def validate_vector(value, dimension: int) -> list[float] | None:
    """Empty published vectors are missing data; malformed nonempty vectors are errors."""
    if value is None or len(value) == 0:
        return None
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected a finite vector of dimension {dimension}; got {vector.shape}.")
    if np.linalg.norm(vector) == 0:
        return None
    return vector.tolist()


def cosine01(left, right) -> float:
    a, b = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Cannot compare vectors from different embedding spaces.")
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not norm:
        return 0.0
    return float(np.clip((np.dot(a, b) / norm + 1) / 2, 0, 1))


class LocalEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = RLock()
        self._queries = OrderedDict()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    cache_folder=str(self.settings.cache_dir / "models"),
                )
                model.max_seq_length = 1024
                if model.get_sentence_embedding_dimension() != self.settings.dimension:
                    raise ValueError("The query model must match the stored Qwen vector dimension.")
                self._model = model
        return self._model

    def documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self.model.encode(
                list(texts),
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

    def query(self, text: str, purpose: str = "tracks") -> tuple[float, ...]:
        instructions = {
            "tracks": "Given a music request, retrieve tracks matching the listener's musical preferences.",
            "profiles": "Given a listener profile, retrieve listeners with similar musical preferences.",
            "goals": "Given a music listening goal, retrieve similar conversation goals.",
        }
        with self._lock:
            key = (text, purpose)
            if key not in self._queries:
                prompt = f"Instruct: {instructions[purpose]}\nQuery: {text}"
                self._queries[key] = tuple(self.documents([prompt])[0])
                if len(self._queries) > 256:
                    self._queries.popitem(last=False)
            self._queries.move_to_end(key)
            return self._queries[key]
