"""Test whether musical-culture similarity improves the saved BPR neighbor pool.

This is a leakage-aware, top-30-neighbor experiment, not a replacement for the
runtime full-pool search. The held-out session profile is available as query
context; neighbor profiles come only from sessions outside held-out targets.
Tune score weights on half the internal-validation users and report the selected
configuration on the other half. No official-test targets or catalog indexes are
read or changed.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STOPWORDS = {"culture", "music", "musical"}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 0.9)


def culture_tokens(value: object) -> frozenset[str]:
    if pd.isna(value):
        return frozenset()
    return frozenset(re.findall(r"[a-z0-9]+", str(value).casefold())) - STOPWORDS


def culture_match(query: object, neighbor: object, method: str) -> float:
    a, b = culture_tokens(query), culture_tokens(neighbor)
    if not a or not b:
        return 0.0
    if method == "exact":
        return float(a == b)
    return len(a & b) / len(a | b)


def load_inputs(data_dir: Path):
    sessions = pd.read_parquet(
        data_dir / "sessions.parquet",
        columns=["session_id", "user_id", "session_date", "source_split",
                 "preferred_musical_culture", "preferred_language"],
    )
    history = pd.read_parquet(
        data_dir / "internal_train_history.parquet",
        columns=["session_id", "user_id", "track_id", "is_positive", "is_labeled"],
    )
    targets = pd.read_parquet(
        data_dir / "internal_validation_targets.parquet",
        columns=["session_id", "user_id", "track_id", "is_positive", "is_labeled"],
    )
    neighbors = pd.read_parquet(
        data_dir / "user_neighbors.parquet",
        columns=["query_user_id", "query_embedding_split", "neighbor_user_id",
                 "neighbor_rank", "cosine_similarity", "neighbor_pool"],
    )
    if not (history.is_positive & history.is_labeled).all():
        raise ValueError("Internal history contains ineligible feedback")
    if not (targets.is_positive & targets.is_labeled).all():
        raise ValueError("Validation targets contain ineligible feedback")
    target_sessions = set(targets.session_id)
    if target_sessions & set(history.session_id):
        raise ValueError("Held-out session appears in training history")
    if sessions.preferred_language.nunique(dropna=False) != 1:
        raise ValueError("Language is no longer constant; review whether to test it")
    return sessions, history, targets, neighbors


def prepare_users(sessions, history, targets, neighbors):
    target_sessions = set(targets.session_id)
    query_profiles = sessions[sessions.session_id.isin(target_sessions)]
    if query_profiles.user_id.duplicated().any():
        raise ValueError("Expected one held-out profile session per validation user")
    query_culture = dict(zip(query_profiles.user_id, query_profiles.preferred_musical_culture))

    # Never use a held-out session as a neighbor's profile, even when that
    # neighbor also happens to be a validation query user.
    allowed = sessions[
        sessions.source_split.eq("train") & ~sessions.session_id.isin(target_sessions)
    ].sort_values(["session_date", "session_id"])
    neighbor_profiles = allowed.drop_duplicates("user_id", keep="last")
    neighbor_culture = dict(
        zip(neighbor_profiles.user_id, neighbor_profiles.preferred_musical_culture)
    )

    history_tracks = {
        user_id: set(group.track_id)
        for user_id, group in history.groupby("user_id")
    }
    relevant = {
        user_id: set(group.track_id)
        for user_id, group in targets.groupby("user_id")
    }
    train_neighbors = neighbors[
        neighbors.query_embedding_split.eq("train")
        & neighbors.query_user_id.isin(relevant)
    ]
    if not train_neighbors.neighbor_pool.eq("train_embeddings_only").all():
        raise ValueError("Neighbor pool includes non-training users")
    if train_neighbors.duplicated(["query_user_id", "neighbor_user_id"]).any():
        raise ValueError("Duplicate neighbor for a validation query")
    if set(train_neighbors.query_user_id) != set(relevant):
        raise ValueError("Some validation users lack saved BPR neighbors")

    by_user = {}
    for user_id, group in train_neighbors.groupby("query_user_id", sort=False):
        query = query_culture[user_id]
        pair_rows = []
        for row in group.itertuples(index=False):
            if row.neighbor_user_id == user_id:
                continue
            candidate_tracks = history_tracks.get(row.neighbor_user_id)
            if not candidate_tracks:
                continue
            culture = neighbor_culture.get(row.neighbor_user_id)
            pair_rows.append(
                (
                    float(row.cosine_similarity),
                    culture_match(query, culture, "exact"),
                    culture_match(query, culture, "token"),
                    candidate_tracks,
                )
            )
        by_user[user_id] = pair_rows
    return by_user, history_tracks, relevant


def top_tracks(pairs, seen: set[str], weight: float, method: str) -> list[str]:
    scores = defaultdict(float)
    for cosine, exact, token, tracks in pairs:
        culture = exact if method == "exact" else token
        similarity = weight * cosine + (1 - weight) * culture
        if similarity <= 0:
            continue
        for track_id in tracks - seen:
            scores[track_id] += similarity
    return sorted(scores, key=lambda track_id: (-scores[track_id], track_id))[:20]


def score(predicted: list[str], relevant: set[str]) -> dict[str, float]:
    hits = [track_id in relevant for track_id in predicted]
    discount = [1 / np.log2(rank + 2) for rank in range(20)]
    dcg = sum(discount[rank] for rank, hit in enumerate(hits) if hit)
    ideal = sum(discount[: min(len(relevant), 20)])
    count = sum(hits)
    return {
        "recall@20": count / len(relevant),
        "ndcg@20": dcg / ideal,
        "hit_rate@20": float(count > 0),
    }


def mean_metrics(rows: dict[str, dict], users: list[str]) -> dict[str, float]:
    return {
        metric: float(np.mean([rows[user_id][metric] for user_id in users]))
        for metric in ("recall@20", "ndcg@20", "hit_rate@20")
    }


def paired_interval(base: dict, candidate: dict, users: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(users), size=(2000, len(users)))
    result = {}
    for metric in ("recall@20", "ndcg@20", "hit_rate@20"):
        differences = np.array(
            [candidate[user_id][metric] - base[user_id][metric] for user_id in users]
        )
        low, high = np.quantile(differences[samples].mean(axis=1), [0.025, 0.975])
        result[metric] = {
            "mean_delta": float(differences.mean()),
            "bootstrap_95ci": [float(low), float(high)],
        }
    return result


def run(data_dir: Path, output: Path, seed: int = 42) -> dict:
    sessions, history, targets, neighbors = load_inputs(data_dir)
    by_user, history_tracks, relevant = prepare_users(sessions, history, targets, neighbors)
    users = sorted(relevant)
    random.Random(seed).shuffle(users)
    tuning, evaluation = users[: len(users) // 2], users[len(users) // 2 :]
    configurations = [("bpr", 1.0, "exact")]
    configurations.extend(
        (method, weight, method)
        for method in ("exact", "token")
        for weight in WEIGHTS
    )
    scored = {}
    for name, weight, method in configurations:
        label = "bpr" if name == "bpr" else f"{name}_bpr_weight_{weight:g}"
        scored[label] = {
            user_id: score(
                top_tracks(by_user[user_id], history_tracks.get(user_id, set()), weight, method),
                relevant[user_id],
            )
            for user_id in users
        }
    baseline = scored["bpr"]
    # Select by tuning NDCG; the unmodified BPR baseline is a valid winner.
    options = list(scored)
    selected = max(
        options,
        key=lambda label: (
            mean_metrics(scored[label], tuning)["ndcg@20"],
            1.0 if label == "bpr" else float(label.rsplit("_", 1)[-1]),
        ),
    )
    profile_pairs = [pair for pairs in by_user.values() for pair in pairs]
    report = {
        "protocol": "saved-top30-bpr-neighbors-internal-temporal-profile-ablation-v1",
        "seed": seed,
        "users": {"tuning": len(tuning), "evaluation": len(evaluation)},
        "feature_check": {
            "preferred_language_unique_values": int(sessions.preferred_language.nunique()),
            "preferred_musical_culture_unique_values": int(sessions.preferred_musical_culture.nunique()),
            "language_used": False,
            "neighbor_pairs_with_eligible_history": len(profile_pairs),
            "fraction_with_exact_culture_match": float(
                np.mean([pair[1] > 0 for pair in profile_pairs])
            ),
            "fraction_with_token_culture_overlap": float(
                np.mean([pair[2] > 0 for pair in profile_pairs])
            ),
        },
        "baseline": {
            "all_users": mean_metrics(baseline, users),
            "tuning": mean_metrics(baseline, tuning),
            "evaluation": mean_metrics(baseline, evaluation),
        },
        "tuning_results": {
            label: mean_metrics(rows, tuning) for label, rows in scored.items()
        },
        "selected_configuration": selected,
        "selected_evaluation": mean_metrics(scored[selected], evaluation),
        "selected_vs_baseline": paired_interval(baseline, scored[selected], evaluation, seed),
        "limitations": [
            "Reweights only the saved top-30 BPR neighbors; cannot discover profile-similar users outside that pool.",
            "Interaction labels are synthetic goal-progress feedback, not confirmed listens or likes.",
            "This is internal validation; do not tune on official test targets.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "eda" / "profile_neighbor_experiment.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
