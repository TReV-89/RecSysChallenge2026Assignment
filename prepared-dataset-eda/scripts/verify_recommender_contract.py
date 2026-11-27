"""Check the prepared tables used by the recommender without training or indexing."""

import json
from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parents[1] / "data"
INTERACTION_KEY = ["session_id", "music_turn", "track_id"]


def load(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet", columns=columns)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    sessions = load("sessions", ["session_id", "user_id", "source_split"])
    interactions = load(
        "conversation_interactions",
        [*INTERACTION_KEY, "user_id", "source_split", "is_labeled", "is_positive"],
    )
    history = load(
        "internal_train_history",
        [*INTERACTION_KEY, "user_id", "session_date", "source_split", "is_labeled", "is_positive"],
    )
    targets = load(
        "internal_validation_targets",
        [*INTERACTION_KEY, "user_id", "session_date", "source_split", "is_labeled", "is_positive"],
    )
    neighbors = load("user_neighbors", ["neighbor_user_id", "neighbor_pool"])
    catalog = load("track_metadata_clean", ["track_id"])

    train_users = set(sessions.loc[sessions.source_split == "train", "user_id"])
    train_positive = interactions.loc[
        (interactions.source_split == "train") & interactions.is_positive
    ]
    require(train_positive.is_labeled.all(), "Train-positive evidence contains unlabeled rows")
    require(history.is_positive.all() and history.is_labeled.all(), "History is not positive and labeled")
    require((history.source_split == "train").all(), "History contains non-training sessions")
    require(targets.is_positive.all() and targets.is_labeled.all(), "Targets are not positive and labeled")
    require((targets.source_split == "train").all(), "Targets contain non-training sessions")
    require(set(history.user_id) <= train_users, "History contains non-training users")
    require(set(neighbors.neighbor_user_id) <= train_users, "Neighbor pool contains non-training users")
    require(
        neighbors.neighbor_pool.eq("train_embeddings_only").all(),
        "Precomputed neighbor pool is not training-only",
    )
    require(
        not set(history.session_id).intersection(targets.session_id),
        "Validation target sessions appear in internal history",
    )
    require(
        history[INTERACTION_KEY].merge(targets[INTERACTION_KEY], on=INTERACTION_KEY).shape[0] == 0,
        "Validation target interactions appear in internal history",
    )
    require(
        len(history[INTERACTION_KEY].merge(train_positive[INTERACTION_KEY], on=INTERACTION_KEY))
        == len(history),
        "Internal history has interactions outside train-positive evidence",
    )
    require(
        set(train_positive.track_id) <= set(catalog.track_id),
        "Train-positive evidence references tracks outside the catalog",
    )

    last_history = history.groupby("user_id").session_date.max()
    first_target = targets.groupby("user_id").session_date.min()
    common = last_history.index.intersection(first_target.index)
    require(len(common) == targets.user_id.nunique(), "Some validation users have no history")
    require(
        (last_history.loc[common] < first_target.loc[common]).all(),
        "Some internal history is not earlier than its validation target",
    )

    print(
        json.dumps(
            {
                "official_test_evidence": "train positive rows in conversation_interactions",
                "official_test_train_positive_rows": len(train_positive),
                "official_test_train_positive_user_track_pairs": len(
                    train_positive[["user_id", "track_id"]].drop_duplicates()
                ),
                "internal_validation_evidence": "internal_train_history only",
                "internal_validation_history_rows": len(history),
                "internal_validation_target_rows_excluded": len(targets),
                "internal_validation_users_with_earlier_history": len(common),
                "neighbor_pool": "training users only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
