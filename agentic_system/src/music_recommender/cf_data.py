"""Import eligible conversational evidence and held-out query vectors for CF retrieval."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import Settings
from .embeddings import validate_vector
from .ingest import stream_dataset
from .store import CatalogStore

HISTORY_KINDS = {
    "official_test": "cf_history_official_test",
    "internal_validation": "cf_history_internal_validation",
}


def load_history(store: CatalogStore, prepared_dir: Path, mode: str) -> dict[str, int]:
    """Store positive training pseudo-interactions, never held-out target rows."""
    if mode not in HISTORY_KINDS:
        raise ValueError(f"Unknown CF history mode: {mode}")
    store.set_state(f"{HISTORY_KINDS[mode]}_ready", False)
    name = (
        "conversation_interactions.parquet"
        if mode == "official_test"
        else "internal_train_history.parquet"
    )
    frame = pd.read_parquet(
        prepared_dir / name,
        columns=["user_id", "session_id", "track_id", "source_split", "is_positive", "is_labeled"],
    )
    if mode == "official_test":
        frame = frame.loc[
            (frame.source_split == "train") & frame.is_positive & frame.is_labeled
        ]
    elif not (
        frame.source_split.eq("train").all()
        and frame.is_positive.all()
        and frame.is_labeled.all()
    ):
        raise ValueError("Internal validation history contains ineligible interactions")
    if mode == "internal_validation":
        targets = pd.read_parquet(prepared_dir / "internal_validation_targets.parquet", columns=["session_id"])
        if frame.session_id.isin(targets.session_id).any():
            raise ValueError("Internal validation targets leaked into history")
    if frame[["user_id", "track_id"]].isna().any().any():
        raise ValueError("Eligible interactions contain missing user or track IDs")
    pairs = frame[["user_id", "track_id"]].drop_duplicates()
    groups = pairs.groupby("user_id")["track_id"].agg(lambda values: sorted(values)).to_dict()
    records = [{"user_id": user_id, "track_ids": track_ids} for user_id, track_ids in groups.items()]
    store.replace_records(HISTORY_KINDS[mode], records, "user_id")
    store.set_state(f"{HISTORY_KINDS[mode]}_ready", True)
    return {"rows": len(frame), "user_track_pairs": len(pairs), "users": len(records)}


def load_test_query_vectors(store: CatalogStore, loader=stream_dataset) -> dict[str, int]:
    """Import published test vectors separately from the training neighbor pool."""
    store.set_state("cf_query_vectors_ready", False)
    records = {}
    counts = {"test_warm": 0, "test_cold": 0, "missing": 0}
    for split in ("test_warm", "test_cold"):
        for row in loader(f"{split}_user_vectors", None):
            vector = validate_vector(row.get("cf-bpr"), 128)
            if vector is None:
                counts["missing"] += 1
                continue
            user_id = str(row["user_id"])
            records[user_id] = {"user_id": user_id, "vector": vector, "split": split}
            counts[split] += 1
    store.replace_records("cf_query_vectors", list(records.values()), "user_id")
    store.set_state("cf_query_vectors_ready", True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", choices=["catalog", "sample"], default="catalog")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "prepared-dataset-eda" / "data",
    )
    parser.add_argument("--mode", choices=list(HISTORY_KINDS), default="official_test")
    parser.add_argument("--skip-test-vectors", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.data_dir /= args.catalog
    store = CatalogStore(settings)
    if not store.state("ready"):
        raise RuntimeError("Ingest the catalog before loading collaborative data")
    report = {"history": load_history(store, args.prepared_dir, args.mode)}
    if not args.skip_test_vectors:
        report["query_vectors"] = load_test_query_vectors(
            store,
            loader=lambda key, unused: stream_dataset(key, settings.cache_dir / "datasets"),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
