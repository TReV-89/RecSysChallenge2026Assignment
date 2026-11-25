from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
OUT = BASE / "eda" / "eda_summary.json"


def load(name: str) -> pd.DataFrame:
    return pq.read_table(DATA / name).to_pandas()


def clean_records(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[dict]:
    result = []
    for row in frame[columns].to_dict("records"):
        cleaned = {}
        for key, value in row.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, (np.integer, int)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                cleaned[key] = round(float(value), digits)
            else:
                cleaned[key] = str(value)
        result.append(cleaned)
    return result


sessions = load("sessions.parquet")
interactions = load("conversation_interactions.parquet")
tracks = load("track_metadata_clean.parquet")
neighbors = load("user_neighbors.parquet")
official_candidates = load("official_test_candidates.parquet")
validation_metrics = load("internal_validation_metrics_by_user.parquet")
subgroups = pd.read_csv(DATA / "subgroup_metrics.csv")
quality = json.loads((DATA / "data_quality_report.json").read_text(encoding="utf-8"))

sessions["session_date"] = pd.to_datetime(sessions["session_date"], errors="coerce")
interactions["session_date"] = pd.to_datetime(interactions["session_date"], errors="coerce")

overview = {
    "sessions": int(len(sessions)),
    "users": int(sessions["user_id"].nunique()),
    "tracks_catalog": int(tracks["track_id"].nunique()),
    "music_turns": int(len(interactions)),
    "labeled_turns": int(interactions["is_labeled"].sum()),
    "positive_turns": int(interactions["is_positive"].sum()),
    "positive_rate_labeled": round(float(interactions.loc[interactions["is_labeled"], "is_positive"].mean()), 4),
    "catalog_coverage": round(float(quality["collaborative_feasibility"]["catalog_coverage_fraction"]), 4),
    "sparsity": round(float(quality["collaborative_feasibility"]["user_item_sparsity"]), 6),
    "shared_user_fraction": round(float(quality["collaborative_feasibility"]["fraction_positive_users_with_shared_track"]), 4),
}

session_split = sessions.groupby("source_split").agg(sessions=("session_id", "size"), users=("user_id", "nunique")).reset_index()
user_split = sessions.groupby("user_split")["user_id"].nunique().sort_values(ascending=False).reset_index(name="users")
year_counts = sessions.assign(year=sessions["session_date"].dt.year).groupby(["year", "source_split"]).size().reset_index(name="sessions")

turn_feedback = (
    interactions[interactions["is_labeled"]]
    .groupby("music_turn")
    .agg(labeled=("track_id", "size"), positives=("is_positive", "sum"))
    .reset_index()
)
turn_feedback["positive_rate"] = turn_feedback["positives"] / turn_feedback["labeled"]

goal_stats = (
    interactions[interactions["is_labeled"]]
    .merge(sessions[["session_id", "goal_category"]], on="session_id", how="left", suffixes=("", "_session"))
    .groupby("goal_category")
    .agg(labeled=("track_id", "size"), positives=("is_positive", "sum"), users=("user_id", "nunique"))
    .reset_index()
)
goal_stats["positive_rate"] = goal_stats["positives"] / goal_stats["labeled"]
goal_stats = goal_stats.sort_values("positive_rate", ascending=False)

age_stats = (
    interactions[interactions["is_labeled"]]
    .merge(sessions[["session_id", "age_group"]], on="session_id", how="left")
    .groupby("age_group")
    .agg(labeled=("track_id", "size"), positives=("is_positive", "sum"), users=("user_id", "nunique"))
    .reset_index()
)
age_stats["positive_rate"] = age_stats["positives"] / age_stats["labeled"]
age_order = ["10s", "20s", "30s", "40s", "50s", "60s"]
age_stats["order"] = age_stats["age_group"].map({v: i for i, v in enumerate(age_order)})
age_stats = age_stats.sort_values("order")

gender_stats = (
    interactions[interactions["is_labeled"]]
    .merge(sessions[["session_id", "gender"]], on="session_id", how="left")
    .groupby("gender")
    .agg(labeled=("track_id", "size"), positives=("is_positive", "sum"), users=("user_id", "nunique"))
    .reset_index()
)
gender_stats["positive_rate"] = gender_stats["positives"] / gender_stats["labeled"]

country_stats = sessions.groupby("country_name")["user_id"].nunique().sort_values(ascending=False).head(12).reset_index(name="users")

sessions_per_user = sessions.groupby("user_id")["session_id"].nunique()
positive_user_tracks = interactions[(interactions["source_split"] == "train") & interactions["is_positive"]][["user_id", "track_id"]].drop_duplicates().groupby("user_id").size()

def binned_counts(values: pd.Series, bins: list[float], labels: list[str]) -> list[dict]:
    binned = pd.cut(values, bins=bins, labels=labels, right=True, include_lowest=True)
    counts = binned.value_counts(sort=False)
    return [{"bin": str(label), "users": int(count)} for label, count in counts.items()]

user_activity = {
    "session_quantiles": {str(k): float(v) for k, v in sessions_per_user.quantile([0, 0.5, 0.9, 0.99, 1]).items()},
    "positive_track_quantiles": {str(k): float(v) for k, v in positive_user_tracks.quantile([0, 0.5, 0.9, 0.99, 1]).items()},
    "sessions_histogram": binned_counts(sessions_per_user, [0, 1, 2, 3, 5, 10, np.inf], ["1", "2", "3", "4-5", "6-10", "11+"]),
    "positive_tracks_histogram": binned_counts(positive_user_tracks, [0, 1, 3, 5, 10, 20, np.inf], ["1", "2-3", "4-5", "6-10", "11-20", "21+"]),
}

positive_train = interactions[(interactions["source_split"] == "train") & interactions["is_positive"]]
track_positive_counts = positive_train.groupby("track_id").size().sort_values(ascending=False)
unique_positive_tracks = int(len(track_positive_counts))
frequency_bins = pd.cut(track_positive_counts, bins=[0, 1, 3, 10, 30, np.inf], labels=["1", "2-3", "4-10", "11-30", "31+"], include_lowest=True)
track_frequency = frequency_bins.value_counts(sort=False).reset_index()
track_frequency.columns = ["frequency_bin", "tracks"]

def top_share(fraction: float) -> float:
    n = max(1, int(np.ceil(len(track_positive_counts) * fraction)))
    return float(track_positive_counts.head(n).sum() / track_positive_counts.sum())

tag_counter = Counter()
for values in tracks["tags_normalized"]:
    tag_counter.update(set(values if values is not None else []))
top_tags = pd.DataFrame(tag_counter.most_common(15), columns=["tag", "tracks"])

positive_artist = (
    positive_train.dropna(subset=["artist_name_primary"])
    .groupby("artist_name_primary")
    .size()
    .sort_values(ascending=False)
    .head(12)
    .reset_index(name="positive_turns")
)

popularity_comparison = (
    interactions[interactions["is_labeled"]]
    .assign(outcome=lambda x: np.where(x["is_positive"], "Positive", "Negative"))
    .groupby("outcome")["popularity"]
    .agg(["count", "mean", "median"])
    .reset_index()
)

metadata_quality = {
    "missing_track_join_rows": int(interactions["track_name_primary"].isna().sum()),
    "missing_track_join_fraction": round(float(interactions["track_name_primary"].isna().mean()), 6),
    "catalog_missing_track_name": int(tracks["track_name_primary"].isna().sum()),
    "catalog_missing_artist_name": int(tracks["artist_name_primary"].isna().sum()),
    "catalog_missing_popularity": int(tracks["popularity"].isna().sum()),
    "catalog_zero_normalized_tags": int((tracks["tag_count_normalized"] == 0).sum()),
    "raw_tag_entries": int(tracks["tag_count_raw"].sum()),
    "normalized_unique_tag_entries": int(tracks["tag_count_normalized"].sum()),
    "invalid_session_dates": int(sessions["session_date"].isna().sum()),
    "duplicate_session_ids": int(sessions["session_id"].duplicated().sum()),
    "duplicate_catalog_track_ids": int(tracks["track_id"].duplicated().sum()),
    "unlabeled_turns": int((~interactions["is_labeled"]).sum()),
    "unlabeled_turns_are_final_turns": bool((interactions.loc[~interactions["is_labeled"], "music_turn"] == 8).all()),
}

candidate_stats = {
    "test_users": int(official_candidates["query_user_id"].nunique()),
    "min_candidates_per_user": int(official_candidates.groupby("query_user_id").size().min()),
    "max_candidates_per_user": int(official_candidates.groupby("query_user_id").size().max()),
    "candidate_partitions": official_candidates.groupby("candidate_partition")["query_user_id"].nunique().to_dict(),
    "neighbor_query_users": neighbors.groupby("query_embedding_split")["query_user_id"].nunique().to_dict(),
}

evaluation = {
    "similar_user": quality["internal_temporal_evaluation"]["similar_user"],
    "popularity_baseline": quality["internal_temporal_evaluation"]["popularity_baseline"],
    "subgroups": clean_records(subgroups, ["group_dimension", "group_value", "users", "precision@20", "recall@20", "ndcg@20", "hit_rate@20"]),
}

summary = {
    "overview": overview,
    "session_split": clean_records(session_split, ["source_split", "sessions", "users"]),
    "user_split": clean_records(user_split, ["user_split", "users"]),
    "year_counts": clean_records(year_counts, ["year", "source_split", "sessions"]),
    "turn_feedback": clean_records(turn_feedback, ["music_turn", "labeled", "positives", "positive_rate"]),
    "goal_stats": clean_records(goal_stats, ["goal_category", "users", "labeled", "positives", "positive_rate"]),
    "age_stats": clean_records(age_stats, ["age_group", "users", "labeled", "positives", "positive_rate"]),
    "gender_stats": clean_records(gender_stats, ["gender", "users", "labeled", "positives", "positive_rate"]),
    "country_stats": clean_records(country_stats, ["country_name", "users"]),
    "user_activity": user_activity,
    "catalog": {
        "unique_positive_tracks": unique_positive_tracks,
        "track_frequency": clean_records(track_frequency, ["frequency_bin", "tracks"]),
        "top_1pct_interaction_share": round(top_share(0.01), 4),
        "top_5pct_interaction_share": round(top_share(0.05), 4),
        "top_10pct_interaction_share": round(top_share(0.10), 4),
        "top_tags": clean_records(top_tags, ["tag", "tracks"]),
        "top_positive_artists": clean_records(positive_artist, ["artist_name_primary", "positive_turns"]),
        "popularity_outcome": clean_records(popularity_comparison, ["outcome", "count", "mean", "median"], digits=2),
    },
    "metadata_quality": metadata_quality,
    "candidate_stats": candidate_stats,
    "evaluation": evaluation,
    "leakage_checks": quality["leakage_checks"],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
