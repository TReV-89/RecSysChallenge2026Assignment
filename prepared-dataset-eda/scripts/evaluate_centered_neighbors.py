"""Compare cosine with training-population-centered cosine on enhanced CF users.

The adapter was fitted without internal-validation users. This experiment uses
its saved weights, identical query/neighbor vectors and eligible history for
both similarity methods. The center is computed from training user vectors
only; held-out target tracks never enter it.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate_culture_adapter import (
    ARTIFACT,
    CATALOG,
    DATA,
    ROOT,
    adjusted,
    interval,
    load_data,
    load_vectors,
    means,
    metrics,
    token_matrix,
    top_indices,
)

OUTPUT = ROOT / "eda" / "centered_neighbor_experiment.json"


def neighbor_scores(
    users: list[str], query: np.ndarray, training: np.ndarray,
    training_ids: list[str], tracks: set[str], positives: dict, relevant: dict,
    center: np.ndarray | None,
) -> dict[str, dict[str, float]]:
    """Use the same full-pool top-30 feedback aggregation for either metric."""
    if center is not None:
        query = query - center
        training = training - center
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    training = training / np.maximum(np.linalg.norm(training, axis=1, keepdims=True), 1e-12)
    similarities = query @ training.T
    self_index = {user: index for index, user in enumerate(training_ids)}
    result = {}
    for row, user in enumerate(users):
        if user in self_index:
            similarities[row, self_index[user]] = -np.inf
        nearest = top_indices(similarities[row], min(31, len(training_ids)))
        seen = positives.get(user, set())
        support = defaultdict(float)
        kept = 0
        for index in nearest:
            similarity = float(similarities[row, index])
            if similarity <= 0 or kept >= 30:
                break
            neighbor_tracks = positives.get(training_ids[index])
            if not neighbor_tracks:
                continue
            kept += 1
            for track in neighbor_tracks - seen:
                if track in tracks:
                    support[track] += similarity
        predicted = sorted(support, key=lambda track: (-support[track], track))[:20]
        result[user] = metrics(predicted, relevant[user])
    return result


def run(data_dir: Path, catalog: Path, adapter: Path, output: Path, seed: int = 42):
    positives, relevant, train_profile, query_profile = load_data(data_dir)
    user_ids, base_users = load_vectors(catalog, "users_bpr")
    track_ids, _ = load_vectors(catalog, "tracks_bpr")
    saved = np.load(adapter, allow_pickle=False)
    vocabulary = {str(token): index for index, token in enumerate(saved["tokens"])}
    alpha = float(saved["user_to_user_alpha"])
    user_index = {user: index for index, user in enumerate(user_ids)}
    users = sorted(relevant)
    if not set(users) <= set(user_index):
        raise ValueError("Validation users lack published CF-BPR factors")
    random.Random(seed).shuffle(users)
    tuning, evaluation = users[: len(users) // 2], users[len(users) // 2 :]
    train_features = token_matrix(user_ids, train_profile, vocabulary)
    query_features = token_matrix(users, query_profile, vocabulary)
    weights = saved["correction_weights"]
    training = adjusted(base_users, train_features, weights, alpha)
    query = adjusted(base_users[[user_index[user] for user in users]], query_features, weights, alpha)
    center = training.mean(axis=0)
    raw = neighbor_scores(
        users, query, training, user_ids, set(track_ids), positives, relevant, None
    )
    centered = neighbor_scores(
        users, query, training, user_ids, set(track_ids), positives, relevant, center
    )
    report = {
        "protocol": "enhanced-cf-users-population-centered-cosine-v1",
        "seed": seed,
        "adapter": str(adapter),
        "center_source": "mean of enhanced training-user factors only",
        "formula": "cos(query-enhanced minus center, neighbor-enhanced minus center)",
        "users": {"tuning": len(tuning), "evaluation": len(evaluation)},
        "training_pool": len(user_ids),
        "alpha": alpha,
        "tuning": {"cosine": means(raw, tuning), "centered_cosine": means(centered, tuning)},
        "evaluation": {"cosine": means(raw, evaluation),
                       "centered_cosine": means(centered, evaluation)},
        "centered_vs_cosine_evaluation": interval(raw, centered, evaluation, seed),
        "limitations": [
            "This is population-centered cosine on latent factors, not classical rating-adjusted cosine.",
            "The result covers the user-to-user candidate route, not the full agent.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--adapter", type=Path, default=ARTIFACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.catalog, args.adapter, args.output, args.seed), indent=2))
