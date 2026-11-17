"""Challenge retrieval/diversity metrics and an explicitly unofficial judge composite.

References: github.com/nlp4musa/music-crs-evaluator (metrics and evaluate_devset.py).
Distinct-2 lowercases and splits on whitespace, retaining punctuation. Bigrams do
not cross response boundaries. Diversity is corpus-wide, not averaged per turn.
"""

import math
from collections import Counter, defaultdict
from itertools import pairwise
from statistics import mean


def ndcg(target: str, predictions: list[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive.")
    if len(set(predictions)) != len(predictions):
        raise ValueError("Predicted track IDs must be unique.")
    return next(
        (1 / math.log2(rank + 1) for rank, item in enumerate(predictions[:k], 1) if item == target),
        0.0,
    )


def distinct2(responses: list[str]) -> float:
    counts = Counter()
    for response in responses:
        tokens = response.lower().split()
        counts.update(pairwise(tokens))
    return len(counts) / sum(counts.values()) if counts else 0.0


def prediction_key(row: dict):
    return row["session_id"], row["turn_number"]


def validate_predictions(cases, predictions, catalog_ids):
    expected = {case.key: case for case in cases}
    if len(expected) != len(cases):
        raise ValueError("Duplicate evaluation case.")
    actual = {}
    for row in predictions:
        key = prediction_key(row)
        if key in actual or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        if row["user_id"] != expected[key].user_id:
            raise ValueError(f"Wrong user ID for {key}")
        ids = row["predicted_track_ids"]
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError("Predicted track IDs must be a list of strings.")
        if len(ids) > 20 or len(ids) != len(set(ids)) or not set(ids) <= catalog_ids:
            raise ValueError(f"Invalid, duplicate or too many track IDs for {key}")
        if not isinstance(row["predicted_response"], str):
            raise ValueError("Predicted response must be text.")  # noqa: TRY004 - input validation
        actual[key] = row
    return expected, actual


def summarize(cases, predictions, catalog_ids: set[str], judges=None, seed=42):
    """Require complete predictions so missing/failed turns cannot inflate scores."""
    if not cases or not catalog_ids:
        raise ValueError("Cases and catalog must not be empty.")
    expected, actual = validate_predictions(cases, predictions, catalog_ids)
    if set(expected) != set(actual):
        raise ValueError(
            f"Missing {len(set(expected) - set(actual))} predictions; run is incomplete."
        )
    if any(case.target_track_id not in catalog_ids for case in cases):
        raise ValueError("Ground-truth track is outside the official catalog.")
    per_turn, per_session = defaultdict(list), defaultdict(list)
    for case in cases:
        scores = {
            f"ndcg@{k}": ndcg(case.target_track_id, actual[case.key]["predicted_track_ids"], k)
            for k in (1, 10, 20)
        }
        per_turn[case.turn_number].append(scores)
        per_session[case.session_id].append(scores)
    # This is the order used by the public evaluator: sessions within each turn,
    # then turns. Complete eight-turn sessions also give equal session weighting.
    turn_scores = {
        str(turn): {
            name: mean(row[name] for row in rows) for name in ("ndcg@1", "ndcg@10", "ndcg@20")
        }
        for turn, rows in sorted(per_turn.items())
    }
    metrics = {
        name: mean(row[name] for row in turn_scores.values())
        for name in ("ndcg@1", "ndcg@10", "ndcg@20")
    }
    recommended = {track for row in predictions for track in row["predicted_track_ids"]}
    metrics.update(
        catalog_diversity=len(recommended) / len(catalog_ids),
        lexical_diversity=distinct2([row["predicted_response"] for row in predictions]),
    )
    subtotal = (
        0.5 * metrics["ndcg@20"]
        + 0.1 * metrics["catalog_diversity"]
        + 0.1 * metrics["lexical_diversity"]
    )
    judge_score = None
    if judges is not None:
        from .judge import JudgeScore

        if set(judges) != set(expected):
            raise ValueError("Judge scores must cover every evaluated turn.")
        validated = {key: JudgeScore.model_validate(value) for key, value in judges.items()}
        judge_score = mean(
            mean(validated[case.key].normalized for case in cases if case.turn_number == turn)
            for turn in per_turn
        )
    return {
        "sessions": len(per_session),
        "turns": len(cases),
        "total_catalog_size": len(catalog_ids),
        "unique_recommended_tracks": len(recommended),
        **metrics,
        "per_turn": turn_scores,
        "ndcg20_session_bootstrap_95ci": session_interval(per_session, seed),
        "llm_judge_proxy": judge_score,
        "weighted_non_judge_subtotal_out_of_0.70": subtotal,
        "composite_proxy": None if judge_score is None else subtotal + 0.3 * judge_score,
        "official_blind_composite": None,
        "modes": dict(Counter(row.get("mode", "unknown") for row in predictions)),
    }


def session_interval(per_session, seed):
    """Percentile bootstrap of complete session means, not independent turns.

    Descriptive sampling uncertainty only; does not capture provider randomness,
    repeated users across sessions or correct for the finite test population.
    """
    import numpy as np

    values = np.array([mean(row["ndcg@20"] for row in rows) for rows in per_session.values()])
    if len(values) < 2 or np.ptp(values) == 0:
        return None  # Avoid implying certainty when the bootstrap is degenerate.
    rng = np.random.default_rng(seed)
    estimates = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(2000)]
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "resamples": 2000, "unit": "session"}
