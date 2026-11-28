"""Train a small culture correction to frozen CF-BPR users; evaluate offline.

The published track factors stay fixed. Fitting uses only eligible positive
pre-holdout interactions from users outside the internal-validation cohort.
Validation users are split into tuning and untouched evaluation halves.
No production catalog vectors are changed.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CATALOG = ROOT.parent / "agentic_system" / "data" / "catalog" / "chroma"
OUTPUT = ROOT / "eda" / "culture_adapter_experiment.json"
ARTIFACT = ROOT / "eda" / "culture_adapter_weights.npz"
STOPWORDS = {"culture", "music", "musical"}
ALPHAS = (0.0, 0.25, 0.5, 1.0)
METRICS = ("recall@20", "ndcg@20", "hit_rate@20")


def tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return set(re.findall(r"[a-z0-9]+", str(value).casefold())) - STOPWORDS


def load_vectors(path: Path, name: str) -> tuple[list[str], np.ndarray]:
    collection = chromadb.PersistentClient(path=str(path)).get_collection(name)
    ids, arrays = [], []
    for offset in range(0, collection.count(), 500):
        batch = collection.get(limit=500, offset=offset, include=["embeddings"])
        ids.extend(batch["ids"])
        arrays.append(np.asarray(batch["embeddings"], dtype=np.float32))
    values = np.concatenate(arrays)
    order = np.argsort(ids)
    return [ids[index] for index in order], values[order]


def load_data(data_dir: Path):
    sessions = pd.read_parquet(
        data_dir / "sessions.parquet",
        columns=["session_id", "user_id", "session_date", "source_split",
                 "preferred_musical_culture"],
    )
    history = pd.read_parquet(
        data_dir / "internal_train_history.parquet",
        columns=["session_id", "user_id", "track_id", "is_positive", "is_labeled"],
    )
    targets = pd.read_parquet(
        data_dir / "internal_validation_targets.parquet",
        columns=["session_id", "user_id", "track_id", "is_positive", "is_labeled"],
    )
    if not (history.is_positive & history.is_labeled).all():
        raise ValueError("Training history includes ineligible feedback")
    if not (targets.is_positive & targets.is_labeled).all():
        raise ValueError("Targets include ineligible feedback")
    if set(history.session_id) & set(targets.session_id):
        raise ValueError("Held-out sessions leaked into training history")
    query_rows = sessions[sessions.session_id.isin(targets.session_id)]
    if query_rows.user_id.duplicated().any():
        raise ValueError("Expected one held-out session per validation user")
    query_profile = dict(zip(query_rows.user_id, query_rows.preferred_musical_culture))
    train_rows = sessions[
        sessions.source_split.eq("train") & ~sessions.session_id.isin(targets.session_id)
    ].sort_values(["session_date", "session_id"])
    train_rows = train_rows.drop_duplicates("user_id", keep="last")
    train_profile = dict(zip(train_rows.user_id, train_rows.preferred_musical_culture))
    positives = {
        user: set(group.track_id) for user, group in history.groupby("user_id")
    }
    relevant = {
        user: set(group.track_id) for user, group in targets.groupby("user_id")
    }
    if set(relevant) != set(query_profile):
        raise ValueError("Missing query profiles")
    return positives, relevant, train_profile, query_profile


def token_matrix(users: list[str], profiles: dict, vocabulary: dict) -> np.ndarray:
    x = np.zeros((len(users), len(vocabulary)), dtype=np.float32)
    for row, user in enumerate(users):
        indexes = [vocabulary[token] for token in tokens(profiles.get(user)) if token in vocabulary]
        if indexes:
            x[row, indexes] = 1 / np.sqrt(len(indexes))
    return x


def train_adapter(
    user_ids: list[str], user_vectors: np.ndarray, track_ids: list[str],
    track_vectors: np.ndarray, positives: dict, profiles: dict, seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit culture-token embeddings with pairwise positive vs random-track loss."""
    rng = np.random.default_rng(seed)
    user_index = {user: index for index, user in enumerate(user_ids)}
    track_index = {track: index for index, track in enumerate(track_ids)}
    fit_users = sorted(set(positives) & set(profiles) & set(user_index))
    document_frequency = Counter(
        token for user in fit_users for token in tokens(profiles[user])
    )
    vocabulary = {
        token: index for index, token in enumerate(
            sorted(token for token, count in document_frequency.items() if count >= 10)
        )
    }
    if not vocabulary:
        raise ValueError("No sufficiently frequent culture tokens")
    x = token_matrix(fit_users, profiles, vocabulary)
    pairs = [
        (row, track_index[track])
        for row, user in enumerate(fit_users)
        for track in sorted(positives[user])
        if track in track_index and x[row].any()
    ]
    if not pairs:
        raise ValueError("No fitting interactions with culture tokens and track factors")
    pair_user, pair_track = np.asarray(pairs, dtype=np.int64).T
    base = torch.from_numpy(user_vectors[[user_index[user] for user in fit_users]])
    item = torch.from_numpy(track_vectors)
    feature = torch.from_numpy(x)
    adapter = nn.Linear(len(vocabulary), user_vectors.shape[1], bias=False)
    nn.init.zeros_(adapter.weight)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.01)
    user_scale = float(np.median(np.linalg.norm(user_vectors, axis=1)))
    item_scale = float(np.median(np.linalg.norm(track_vectors, axis=1)))
    denominator = user_scale * item_scale
    batch_size = 1024
    losses = []
    torch.manual_seed(seed)
    for _ in range(8):
        order = rng.permutation(len(pairs))
        running = 0.0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            users = torch.from_numpy(pair_user[selected])
            positive = torch.from_numpy(pair_track[selected])
            negative = torch.from_numpy(rng.integers(len(track_ids), size=len(selected)))
            corrected = base[users] + user_scale * adapter(feature[users])
            margin = (corrected * (item[positive] - item[negative])).sum(1) / denominator
            loss = F.softplus(-margin).mean() + 0.01 * adapter.weight.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(selected)
        losses.append(running / len(pairs))
    return adapter.weight.detach().numpy().T * user_scale, {
        "fit_users": len(fit_users),
        "fit_interactions": len(pairs),
        "vocabulary_size": len(vocabulary),
        "vocabulary": vocabulary,
        "training_loss_by_epoch": losses,
        "user_factor_median_norm": user_scale,
        "track_factor_median_norm": item_scale,
    }


def adjusted(base: np.ndarray, x: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    delta = x @ weights
    limit = 0.5 * np.linalg.norm(base, axis=1, keepdims=True)
    delta *= np.minimum(1, limit / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12))
    return base + alpha * delta


def top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    indexes = np.argpartition(-scores, count - 1)[:count]
    return indexes[np.lexsort((indexes, -scores[indexes]))]


def metrics(predicted: list[str], relevant: set[str]) -> dict[str, float]:
    hits = [track in relevant for track in predicted]
    ideal = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), 20)))
    return {
        "recall@20": sum(hits) / len(relevant),
        "ndcg@20": sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits)) / ideal,
        "hit_rate@20": float(any(hits)),
    }


def evaluate(
    users: list[str], query_vectors: np.ndarray, neighbor_vectors: np.ndarray,
    neighbor_ids: list[str], track_vectors: np.ndarray, track_ids: list[str],
    positives: dict, relevant: dict,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate both full-pool user neighbors and direct user-to-item ranking."""
    neighbor_norm = neighbor_vectors / np.maximum(
        np.linalg.norm(neighbor_vectors, axis=1, keepdims=True), 1e-12
    )
    query_norm = query_vectors / np.maximum(
        np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12
    )
    neighbor_scores = query_norm @ neighbor_norm.T
    item_scores = query_vectors @ track_vectors.T
    track_index = {track: index for index, track in enumerate(track_ids)}
    output = {"user_to_user": {}, "user_to_item": {}}
    for row, user in enumerate(users):
        seen = positives.get(user, set())
        for track in seen:
            index = track_index.get(track)
            if index is not None:
                item_scores[row, index] = -np.inf
        top_items = [track_ids[index] for index in top_indices(item_scores[row], 20)]
        output["user_to_item"][user] = metrics(top_items, relevant[user])

        neighbor_scores[row, np.array(neighbor_ids) == user] = -np.inf
        # Search the whole training factor pool, then retain 30 positive neighbors.
        index = top_indices(neighbor_scores[row], min(31, len(neighbor_ids)))
        support = defaultdict(float)
        kept = 0
        for n in index:
            similarity = float(neighbor_scores[row, n])
            if similarity <= 0 or kept >= 30:
                break
            tracks = positives.get(neighbor_ids[n])
            if not tracks:
                continue
            kept += 1
            for track in tracks - seen:
                if track in track_index:
                    support[track] += similarity
        top_neighbor_items = sorted(support, key=lambda track: (-support[track], track))[:20]
        output["user_to_user"][user] = metrics(top_neighbor_items, relevant[user])
    return output


def means(rows: dict, users: list[str]) -> dict[str, float]:
    return {metric: float(np.mean([rows[user][metric] for user in users])) for metric in METRICS}


def interval(base: dict, variant: dict, users: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(users), size=(2000, len(users)))
    result = {}
    for metric in METRICS:
        delta = np.asarray([variant[user][metric] - base[user][metric] for user in users])
        result[metric] = {
            "delta": float(delta.mean()),
            "bootstrap_95ci": list(map(float, np.quantile(delta[samples].mean(1), [0.025, 0.975]))),
        }
    return result


def run(data_dir: Path, catalog: Path, output: Path, artifact: Path, seed: int = 42) -> dict:
    torch.set_num_threads(4)
    positives, relevant, train_profile, query_profile = load_data(data_dir)
    user_ids, user_vectors = load_vectors(catalog, "users_bpr")
    track_ids, track_vectors = load_vectors(catalog, "tracks_bpr")
    user_index = {user: index for index, user in enumerate(user_ids)}
    validation_users = sorted(set(relevant) & set(user_index))
    if set(validation_users) != set(relevant):
        raise ValueError("Some validation users lack a published BPR factor")
    random.Random(seed).shuffle(validation_users)
    tuning = validation_users[: len(validation_users) // 2]
    evaluation = validation_users[len(validation_users) // 2 :]
    fit_positives = {user: tracks for user, tracks in positives.items() if user not in relevant}
    weights, fit = train_adapter(
        user_ids, user_vectors, track_ids, track_vectors, fit_positives, train_profile, seed
    )
    vocabulary = fit.pop("vocabulary")
    query_base = user_vectors[[user_index[user] for user in validation_users]]
    train_x = token_matrix(user_ids, train_profile, vocabulary)
    query_x = token_matrix(validation_users, query_profile, vocabulary)
    scored = {}
    for alpha in ALPHAS:
        query = adjusted(query_base, query_x, weights, alpha)
        neighbors = adjusted(user_vectors, train_x, weights, alpha)
        scored[str(alpha)] = evaluate(
            validation_users, query, neighbors, user_ids, track_vectors, track_ids,
            positives, relevant,
        )
    selected = {}
    for source in ("user_to_user", "user_to_item"):
        selected_alpha = max(
            ALPHAS,
            key=lambda alpha: (means(scored[str(alpha)][source], tuning)["ndcg@20"], -alpha),
        )
        selected[source] = {
            "alpha": selected_alpha,
            "evaluation": means(scored[str(selected_alpha)][source], evaluation),
            "evaluation_vs_baseline": interval(
                scored["0.0"][source], scored[str(selected_alpha)][source], evaluation, seed
            ),
        }
    # A matched control tests whether attaching the learned culture correction
    # to the wrong people preserves the gain. The same trained weights and
    # tuning-selected alpha are used; no control is retuned on evaluation.
    rng = np.random.default_rng(seed + 1)
    shuffled_query_x = query_x[rng.permutation(len(query_x))]
    shuffled_train_x = train_x[rng.permutation(len(train_x))]
    shuffled_control = {}
    for alpha in {value["alpha"] for value in selected.values()}:
        shuffled_control[alpha] = evaluate(
            validation_users,
            adjusted(query_base, shuffled_query_x, weights, alpha),
            adjusted(user_vectors, shuffled_train_x, weights, alpha),
            user_ids, track_vectors, track_ids, positives, relevant,
        )
    for source, result in selected.items():
        control_rows = shuffled_control[result["alpha"]][source]
        result["shuffled_culture_control_evaluation"] = means(control_rows, evaluation)
        result["selected_vs_shuffled_control"] = interval(
            control_rows, scored[str(result["alpha"])][source], evaluation, seed + 1
        )
    report = {
        "protocol": "frozen-track-factor-culture-adapter-v1",
        "seed": seed,
        "fit": fit,
        "split": {"tuning_users": len(tuning), "evaluation_users": len(evaluation)},
        "feature_coverage": {
            "fit_users_with_vocabulary_tokens": int(sum(token_matrix(
                list(fit_positives), train_profile, vocabulary
            ).any(axis=1))),
            "validation_queries_with_vocabulary_tokens": int(query_x.any(axis=1).sum()),
        },
        "tuning": {
            source: {str(alpha): means(scored[str(alpha)][source], tuning) for alpha in ALPHAS}
            for source in ("user_to_user", "user_to_item")
        },
        "baseline_evaluation": {
            source: means(scored["0.0"][source], evaluation)
            for source in ("user_to_user", "user_to_item")
        },
        "selected": selected,
        "adapter_artifact": str(artifact),
        "limitations": [
            "Feedback is synthetic next-turn goal progress, not confirmed listening or liking.",
            "Only preferred_musical_culture is adapted; published BPR track factors are frozen.",
            "Official test targets and production catalog vectors are not changed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact,
        tokens=np.asarray(list(vocabulary)),
        correction_weights=weights,
        user_to_user_alpha=np.float32(selected["user_to_user"]["alpha"]),
        user_to_item_alpha=np.float32(selected["user_to_item"]["alpha"]),
        maximum_correction_fraction=np.float32(0.5),
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.data_dir, args.catalog, args.output, args.artifact, args.seed)
    print(json.dumps({"selected": result["selected"], "fit": result["fit"]}, indent=2))
