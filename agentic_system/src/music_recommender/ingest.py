"""Stream published vectors into Chroma; locally embed only profiles and goals."""

import argparse
import json
from itertools import islice

from .config import Settings
from .embeddings import LocalEmbedder, validate_vector
from .store import CatalogStore

DATASETS = {
    "sessions": ("talkpl-ai/TalkPlayData-Challenge-Dataset", "train"),
    "tracks": ("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", "all_tracks"),
    "users": ("talkpl-ai/TalkPlayData-Challenge-User-Metadata", "all_users"),
    "track_vectors": ("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", "all_tracks"),
    "user_vectors": ("talkpl-ai/TalkPlayData-Challenge-User-Embeddings", "train"),
}


def batches(rows, size=128):
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        yield batch


def profile_text(profile: dict) -> str:
    fields = [
        "preferred_musical_culture",
        "preferred_language",
        "age_group",
        "country_name",
        "gender",
    ]
    return "; ".join(
        f"{key.replace('_', ' ')}: {profile[key]}" for key in fields if profile.get(key)
    )


def session_record(row: dict) -> dict:
    """Keep profile/goal context only; never import conversation answers or hidden thoughts."""
    goal = row.get("conversation_goal") or {}
    profile = row.get("user_profile") or {}
    specificity = goal.get("specificity")
    if specificity not in {None, "HH", "HL", "LH", "LL"}:
        raise ValueError(f"Unexpected specificity label: {specificity}")
    return {
        "session_id": str(row["session_id"]),
        "user_id": str(row["user_id"]),
        "profile": profile,
        "profile_text": profile_text(profile),
        "goal": goal.get("listener_goal") or "",
        "specificity": specificity,
        "category": goal.get("category") or "",
        "split": "train",
    }


def stream_dataset(key, cache_dir):
    from datasets import load_dataset

    repo, split = DATASETS[key]
    return load_dataset(repo, split=split, streaming=True, cache_dir=str(cache_dir))


def ingest(settings: Settings, sample=False, loader=stream_dataset, embedder=None):
    store = CatalogStore(settings)
    scope = {
        "sample": sample,
        "track_limit": 300 if sample else None,
        "session_limit": 24 if sample else None,
    }
    previous = store.state("scope")
    if previous is not None and previous != scope:
        raise ValueError("Sample and full indexes must use different data directories.")
    store.set_state("scope", scope)
    store.set_state("ready", False)
    embedder = embedder or LocalEmbedder(settings)
    report = {"missing_track_text": 0, "missing_track_cf": 0, "missing_user_cf": 0}

    def rows(key):
        return loader(key, settings.cache_dir / "datasets")

    print("Loading training profile and goal context...", flush=True)
    session_rows = rows("sessions")
    if sample:
        session_rows = islice(session_rows, scope["session_limit"])
    sessions = [session_record(row) for row in session_rows]
    if not sessions:
        raise ValueError("The training session dataset is empty.")
    wanted_users = {row["user_id"] for row in sessions}
    for batch in batches(sessions):
        store.save_records("sessions", batch, "session_id")

    print("Storing supplied track embeddings without re-embedding...", flush=True)
    track_rows = rows("track_vectors")
    if sample:
        track_rows = islice(track_rows, scope["track_limit"])
    wanted_tracks = set()
    for batch in batches(track_rows):
        for key, column, dimension, missing in [
            ("tracks", settings.track_embedding_column, 1024, "missing_track_text"),
            ("tracks_cf", "cf-bpr", 128, "missing_track_cf"),
        ]:
            ids, vectors, metas = [], [], []
            for row in batch:
                if column not in row:
                    raise ValueError(f"Dataset is missing required column {column!r}.")
                vector = validate_vector(row[column], dimension)
                if vector is None:
                    report[missing] += 1
                    continue
                track_id = str(row["track_id"])
                ids.append(track_id)
                vectors.append(vector)
                metas.append({"track_id": track_id, "source_column": column})
                if key == "tracks":
                    wanted_tracks.add(track_id)
            store.upsert(key, ids, vectors, metas)
        print(f"  {len(wanted_tracks):,} searchable tracks", flush=True)

    print("Joining track metadata by track_id...", flush=True)
    found_tracks = set()
    for batch in batches(rows("tracks")):
        selected = [row for row in batch if str(row["track_id"]) in wanted_tracks]
        store.save_records("tracks", selected, "track_id")
        found_tracks.update(str(row["track_id"]) for row in selected)
        if found_tracks == wanted_tracks:
            break
    if wanted_tracks - found_tracks:
        raise ValueError(f"Missing metadata for {len(wanted_tracks - found_tracks)} track IDs.")

    print("Loading user metadata...", flush=True)
    for batch in batches(rows("users")):
        selected = [row for row in batch if not sample or str(row["user_id"]) in wanted_users]
        store.save_records("users", selected, "user_id")

    print("Storing supplied training user CF vectors in their own collection...", flush=True)
    for batch in batches(rows("user_vectors")):
        ids, vectors, metas = [], [], []
        for row in batch:
            user_id = str(row["user_id"])
            if sample and user_id not in wanted_users:
                continue
            vector = validate_vector(row.get("cf-bpr"), 128)
            if vector is None:
                report["missing_user_cf"] += 1
                continue
            ids.append(user_id)
            vectors.append(vector)
            metas.append({"user_id": user_id, "split": "train", "source_column": "cf-bpr"})
        store.upsert("users_cf", ids, vectors, metas)

    print("Embedding training profiles and conversation goals locally with Qwen...", flush=True)
    for key, field in [("profiles", "profile_text"), ("goals", "goal")]:
        for batch in batches(sessions, settings.embedding_batch_size):
            existing = set(store.vectors(key, [row["session_id"] for row in batch]))
            selected = [row for row in batch if row[field] and row["session_id"] not in existing]
            if not selected:
                continue
            vectors = embedder.documents([row[field] for row in selected])
            store.upsert(
                key,
                [row["session_id"] for row in selected],
                vectors,
                [
                    {
                        "user_id": row["user_id"],
                        "specificity": row["specificity"] or "unknown",
                        "split": "train",
                    }
                    for row in selected
                ],
            )
        print(f"  {key}: {store.collections[key].count():,}", flush=True)
    counts = store.counts()
    if not counts["tracks"] or not counts["profiles"] or not counts["goals"]:
        raise ValueError("Ingestion produced an empty required collection.")
    report.update(counts)
    store.set_state("report", report)
    store.set_state("ready", True)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", action="store_true", help="Build 300-track / 24-session smoke-test index."
    )
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.data_dir:
        from pathlib import Path

        settings.data_dir = Path(args.data_dir).resolve()
    else:
        settings.data_dir /= "sample" if args.sample else "catalog"
    ingest(settings, sample=args.sample)


if __name__ == "__main__":
    main()
