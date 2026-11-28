"""Refit the evaluated culture adapter for internal or official-test retrieval.

The validation experiment chooses the two correction strengths. The default
internal-validation fit excludes held-out sessions and reads only their IDs.
The official-test fit uses all eligible training-split feedback. Neither scope
uses official-test outcomes or changes published CF-BPR factors in the catalog.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluate_culture_adapter import (
    CATALOG,
    DATA,
    ROOT,
    adjusted,
    load_vectors,
    token_matrix,
    train_adapter,
)

EXPERIMENT = ROOT / "eda" / "culture_adapter_experiment.json"
WEIGHTS = ROOT / "eda" / "culture_adapter_final_weights.npz"
USER_VECTORS = ROOT / "eda" / "culture_adapter_final_users.npz"
REPORT = ROOT / "eda" / "culture_adapter_final_fit.json"
OFFICIAL_WEIGHTS = ROOT / "eda" / "culture_adapter_official_weights.npz"
OFFICIAL_USERS = ROOT / "eda" / "culture_adapter_official_users.npz"
OFFICIAL_REPORT = ROOT / "eda" / "culture_adapter_official_fit.json"


def load_training(
    data_dir: Path, scope: str
) -> tuple[dict[str, set[str]], dict[str, str], int]:
    target_sessions = (
        set(pd.read_parquet(
            data_dir / "internal_validation_targets.parquet", columns=["session_id"]
        ).session_id)
        if scope == "internal_validation" else set()
    )
    history = pd.read_parquet(
        data_dir / (
            "internal_train_history.parquet" if scope == "internal_validation"
            else "conversation_interactions.parquet"
        ),
        columns=[
            "session_id", "user_id", "track_id", "source_split", "is_positive", "is_labeled"
        ],
    )
    if scope == "official_test":
        history = history[
            history.source_split.eq("train") & history.is_positive & history.is_labeled
        ]
    if set(history.session_id) & target_sessions:
        raise ValueError("Held-out sessions appear in training history")
    if not (history.is_positive & history.is_labeled).all():
        raise ValueError("Training history includes ineligible feedback")
    sessions = pd.read_parquet(
        data_dir / "sessions.parquet",
        columns=[
            "session_id", "user_id", "session_date", "source_split",
            "preferred_musical_culture",
        ],
    )
    sessions = sessions[
        sessions.source_split.eq("train") & ~sessions.session_id.isin(target_sessions)
    ].sort_values(["session_date", "session_id"])
    profiles = sessions.drop_duplicates("user_id", keep="last")
    culture = dict(zip(profiles.user_id, profiles.preferred_musical_culture))
    positives = {
        user: set(group.track_id) for user, group in history.groupby("user_id")
    }
    return positives, culture, len(target_sessions)


def run(
    data_dir: Path, catalog: Path, experiment: Path, weights_path: Path,
    users_path: Path, report_path: Path, scope: str = "internal_validation", seed: int = 42,
) -> dict:
    if scope not in {"internal_validation", "official_test"}:
        raise ValueError("Unknown training scope")
    torch.set_num_threads(4)
    earlier = json.loads(experiment.read_text(encoding="utf-8"))
    if earlier["protocol"] != "frozen-track-factor-culture-adapter-v1":
        raise ValueError("Unexpected tuning experiment protocol")
    alphas = {
        source: float(earlier["selected"][source]["alpha"])
        for source in ("user_to_user", "user_to_item")
    }
    positives, culture, heldout_sessions = load_training(data_dir, scope)
    user_ids, user_vectors = load_vectors(catalog, "users_bpr")
    track_ids, track_vectors = load_vectors(catalog, "tracks_bpr")
    base_hash = hashlib.sha256(user_vectors.astype("<f4", copy=False).tobytes()).hexdigest()
    correction, fit = train_adapter(
        user_ids, user_vectors, track_ids, track_vectors, positives, culture, seed
    )
    vocabulary = fit.pop("vocabulary")
    features = token_matrix(user_ids, culture, vocabulary)
    adapted_users = {
        source: adjusted(user_vectors, features, correction, alpha)
        for source, alpha in alphas.items()
    }
    report = {
        "protocol": "frozen-track-factor-culture-adapter-final-fit-v1",
        "training_scope": scope,
        "seed": seed,
        "fit": fit,
        "eligible_history_users": len(positives),
        "heldout_sessions_excluded": heldout_sessions,
        "published_training_user_factors": len(user_ids),
        "published_track_factors": len(track_ids),
        "base_training_user_factors_sha256": base_hash,
        "indexed_users_with_culture_tokens": int(features.any(axis=1).sum()),
        "selected_alphas_from_experiment": alphas,
        "tuning_experiment": str(experiment),
        "weights_artifact": str(weights_path),
        "adapted_training_users_artifact": str(users_path),
        "note": (
            "The evaluation scores belong to the earlier nonvalidation-user fit; "
            "this final refit must be checked in the end-to-end pipeline before use."
        ),
    }
    for path in (weights_path, users_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        weights_path,
        tokens=np.asarray(list(vocabulary)),
        correction_weights=correction,
        user_to_user_alpha=np.float32(alphas["user_to_user"]),
        user_to_item_alpha=np.float32(alphas["user_to_item"]),
        maximum_correction_fraction=np.float32(0.5),
    )
    np.savez_compressed(
        users_path,
        user_ids=np.asarray(user_ids),
        base_user_factors_sha256=np.asarray(base_hash),
        user_to_user_vectors=adapted_users["user_to_user"],
        user_to_item_vectors=adapted_users["user_to_item"],
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--scope", choices=["internal_validation", "official_test"],
                        default="internal_validation")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--users", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    defaults = (
        (WEIGHTS, USER_VECTORS, REPORT)
        if args.scope == "internal_validation"
        else (OFFICIAL_WEIGHTS, OFFICIAL_USERS, OFFICIAL_REPORT)
    )
    print(json.dumps(run(
        args.data_dir, args.catalog, args.experiment, args.weights or defaults[0],
        args.users or defaults[1], args.report or defaults[2], args.scope, args.seed,
    ), indent=2))
