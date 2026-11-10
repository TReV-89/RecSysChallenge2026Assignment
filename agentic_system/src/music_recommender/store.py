import json
import sqlite3
from pathlib import Path

from .config import Settings

COLLECTIONS = {
    "tracks": ("tracks_qwen_metadata", 1024, "cosine"),
    "profiles": ("session_profiles_qwen", 1024, "cosine"),
    "goals": ("session_goals_qwen", 1024, "cosine"),
    "tracks_cf": ("tracks_bpr", 128, "ip"),
    "users_cf": ("users_bpr", 128, "ip"),
}


class CatalogStore:
    def __init__(self, settings: Settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = Path(settings.data_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "catalog.sqlite3"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))"
            )
            db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        self.client = chromadb.PersistentClient(
            path=str(self.path / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collections = {
            key: self.client.get_or_create_collection(
                name=name,
                embedding_function=None,
                metadata={"hnsw:space": metric, "dimension": dimension},
            )
            for key, (name, dimension, metric) in COLLECTIONS.items()
        }
        identity = {
            "version": 1,
            "model": settings.embedding_model,
            "track_column": settings.track_embedding_column,
            "dimension": settings.dimension,
        }
        previous = self.state("identity")
        if previous is not None and previous != identity:
            raise ValueError(
                "This index uses a different embedding configuration. Choose a new data directory."
            )
        self.set_state("identity", identity)

    def connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def state(self, key: str):
        with self.connect() as db:
            result = db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(result[0]) if result else None

    def set_state(self, key: str, value):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def save_records(self, kind: str, rows: list[dict], id_field: str):
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO records VALUES (?,?,?)",
                [(kind, row[id_field], json.dumps(row, ensure_ascii=False)) for row in rows],
            )

    def records(self, kind: str, ids: list[str] | None = None) -> dict[str, dict]:
        with self.connect() as db:
            if ids is None:
                rows = db.execute(
                    "SELECT id,payload FROM records WHERE kind=? ORDER BY id", (kind,)
                ).fetchall()
            else:
                rows = []
                for start in range(0, len(ids), 500):
                    batch = ids[start : start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    rows.extend(
                        db.execute(
                            f"SELECT id,payload FROM records WHERE kind=? AND id IN ({placeholders})",
                            [kind, *batch],
                        ).fetchall()
                    )
        return {key: json.loads(payload) for key, payload in rows}

    def upsert(self, kind: str, ids, embeddings, metadata):
        if ids:
            self.collections[kind].upsert(ids=ids, embeddings=embeddings, metadatas=metadata)

    def vectors(self, kind: str, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        result = self.collections[kind].get(ids=list(dict.fromkeys(ids)), include=["embeddings"])
        return dict(zip(result["ids"], result["embeddings"]))

    def search(self, kind: str, vector, limit: int, where=None) -> list[str]:
        collection = self.collections[kind]
        count = collection.count()
        if not count:
            return []
        args = {"query_embeddings": [list(vector)], "n_results": min(limit, count), "include": []}
        if where:
            args["where"] = where
        return collection.query(**args)["ids"][0]

    def counts(self) -> dict[str, int]:
        return {key: collection.count() for key, collection in self.collections.items()}
