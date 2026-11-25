from __future__ import annotations

import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
SUMMARY_PATH = BASE / "eda" / "eda_summary.json"
OUT = BASE / "eda"
FIGURES = OUT / "figures"
PDF_PATH = OUT / "talkplay_eda_report.pdf"

BLUE = "#2878B5"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
PURPLE = "#7B61A8"
GRAY = "#7A7A7A"


def load(name: str) -> pd.DataFrame:
    return pq.read_table(DATA / name).to_pandas()


def add_footer(fig: plt.Figure, page_number: int) -> None:
    fig.text(0.02, 0.018, "TalkPlay prepared dataset - static exploratory data analysis", fontsize=8, color=GRAY)
    fig.text(0.98, 0.018, f"Page {page_number}", ha="right", fontsize=8, color=GRAY)


def save_page(pdf: PdfPages, fig: plt.Figure, page_number: int, filename: str) -> None:
    add_footer(fig, page_number)
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight", facecolor="white")
    pdf.savefig(fig, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax, percent: bool = False, horizontal: bool = False, decimals: int = 1) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            value = bar.get_width() if horizontal else bar.get_height()
            labels.append(f"{100 * value:.{decimals}f}%" if percent else f"{value:,.0f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)


def title(fig: plt.Figure, heading: str, subtitle: str) -> None:
    fig.suptitle(heading, x=0.06, y=0.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.91, subtitle, fontsize=10, color=GRAY)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if FIGURES.exists():
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    sessions = load("sessions.parquet")
    interactions = load("conversation_interactions.parquet")
    tracks = load("track_metadata_clean.parquet")
    sessions["session_date"] = pd.to_datetime(sessions["session_date"], errors="coerce")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "figure.titlesize": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    page = 0
    with PdfPages(PDF_PATH) as pdf:
        page += 1
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig.text(0.07, 0.88, "TalkPlay Dataset", fontsize=30, fontweight="bold", color=BLUE)
        fig.text(0.07, 0.81, "Static Exploratory Data Analysis", fontsize=22, fontweight="bold")
        fig.text(0.07, 0.75, "Sessions, conversational feedback, collaborative evidence, catalog coverage, cold start, and validation", fontsize=11, color=GRAY)
        cards = [
            ("16,199", "sessions"), ("8,720", "unique users"),
            ("47,071", "catalog tracks"), ("129,592", "music turns"),
            ("99.987%", "user-item sparsity"), ("55.5%", "positive catalog coverage"),
        ]
        for idx, (value, label) in enumerate(cards):
            col = idx % 3
            row = idx // 3
            x = 0.07 + col * 0.30
            y = 0.58 - row * 0.18
            fig.text(x, y, value, fontsize=22, fontweight="bold", color=BLUE)
            fig.text(x, y - 0.045, label, fontsize=10, color=GRAY)
        caveat = (
            "Important: the interaction table contains tracks recommended in synthetic conversations and labels them with the next user's goal-progress assessment. "
            "These are conversational pseudo-interactions, not verified historical listening events."
        )
        fig.text(0.07, 0.23, "Interpretation boundary", fontsize=12, fontweight="bold")
        fig.text(0.07, 0.13, "\n".join(textwrap.wrap(caveat, 110)), fontsize=10, color=GRAY, linespacing=1.5)
        save_page(pdf, fig, page, "00_overview.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.35, 1]})
        title(fig, "Dataset structure and user activity", "Train sessions cover 2006-2018; test sessions are dated 2019-2020. Most users have limited repeated-session evidence.")
        timeline = sessions.assign(year=sessions["session_date"].dt.year).groupby(["year", "source_split"]).size().reset_index(name="sessions")
        pivot = timeline.pivot(index="year", columns="source_split", values="sessions").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=axes[0], color=[ORANGE, BLUE] if list(pivot.columns) == ["test", "train"] else [BLUE, ORANGE])
        axes[0].set_title("Sessions by year")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("Sessions")
        axes[0].legend(title="Split")
        sessions_per_user = sessions.groupby("user_id")["session_id"].nunique()
        activity = pd.cut(sessions_per_user, bins=[0, 1, 2, 3, 5, 10, np.inf], labels=["1", "2", "3", "4-5", "6-10", "11+"]).value_counts(sort=False)
        sns.barplot(x=activity.index.astype(str), y=activity.values, ax=axes[1], color=BLUE)
        axes[1].set_title("Sessions per user")
        axes[1].set_xlabel("Sessions per user")
        axes[1].set_ylabel("Users")
        label_bars(axes[1])
        fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.87])
        save_page(pdf, fig, page, "01_structure_and_activity.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1, 1.25]})
        title(fig, "Conversation feedback", "The positive-feedback rate declines after the first turn, while performance differs substantially across discovery goals.")
        turn = pd.DataFrame(summary["turn_feedback"])
        sns.barplot(data=turn, x="music_turn", y="positive_rate", ax=axes[0], color=BLUE)
        axes[0].set_title("Positive feedback rate by music turn")
        axes[0].set_xlabel("Music turn")
        axes[0].set_ylabel("Positive rate")
        axes[0].set_ylim(0, 0.65)
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        label_bars(axes[0], percent=True)
        goals = pd.DataFrame(summary["goal_stats"]).sort_values("positive_rate")
        sns.barplot(data=goals, y="goal_category", x="positive_rate", ax=axes[1], color=GREEN)
        axes[1].set_title("Positive feedback rate by discovery goal")
        axes[1].set_xlabel("Positive rate")
        axes[1].set_ylabel("Goal category")
        axes[1].set_xlim(0, 0.65)
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        label_bars(axes[1], percent=True, horizontal=True)
        fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.87])
        save_page(pdf, fig, page, "02_conversation_feedback.png")

        page += 1
        fig, axes = plt.subplots(1, 3, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1, 0.75, 1.45]})
        title(fig, "User representation", "The sample is concentrated among users in their 20s, men, and a small set of countries. Small older-age groups require cautious interpretation.")
        age = pd.DataFrame(summary["age_stats"])
        sns.barplot(data=age, x="age_group", y="users", ax=axes[0], color=BLUE)
        axes[0].set_title("Users by age group")
        axes[0].set_xlabel("Age group")
        axes[0].set_ylabel("Unique users")
        axes[0].tick_params(axis="x", rotation=25)
        gender = pd.DataFrame(summary["gender_stats"])
        sns.barplot(data=gender, x="gender", y="users", ax=axes[1], palette=[PURPLE, BLUE], hue="gender", legend=False)
        axes[1].set_title("Users by gender")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Unique users")
        countries = pd.DataFrame(summary["country_stats"]).sort_values("users")
        sns.barplot(data=countries, y="country_name", x="users", ax=axes[2], color=ORANGE)
        axes[2].set_title("Largest user populations")
        axes[2].set_xlabel("Unique users")
        axes[2].set_ylabel("")
        fig.tight_layout(rect=[0.03, 0.06, 0.99, 0.87])
        save_page(pdf, fig, page, "03_user_representation.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1, 1.3]})
        title(fig, "Collaborative evidence is sparse and long-tailed", "Only 55.5% of catalog tracks appear in positive train evidence. Most observed positive tracks occur once.")
        freq = pd.DataFrame(summary["catalog"]["track_frequency"])
        sns.barplot(data=freq, x="frequency_bin", y="tracks", ax=axes[0], color=BLUE)
        axes[0].set_title("Positive occurrences per track")
        axes[0].set_xlabel("Positive occurrences")
        axes[0].set_ylabel("Tracks")
        label_bars(axes[0])
        positive_counts = interactions[(interactions["source_split"] == "train") & interactions["is_positive"]].groupby("track_id").size().sort_values(ascending=False).reset_index(drop=True)
        rank = np.arange(1, len(positive_counts) + 1)
        axes[1].plot(rank, positive_counts.values, color=BLUE, linewidth=1.2)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("Track-frequency rank curve")
        axes[1].set_xlabel("Track rank (log scale)")
        axes[1].set_ylabel("Positive occurrences (log scale)")
        axes[1].grid(True, which="both", alpha=0.25)
        fig.text(0.06, 0.08, "52,550 unique positive user-track pairs | 99.987% sparse | 91.5% of positive users share at least one track with another user", fontsize=10, color=GRAY)
        fig.tight_layout(rect=[0.04, 0.10, 0.98, 0.87])
        save_page(pdf, fig, page, "04_sparsity_and_long_tail.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        title(fig, "Catalog content", "Rock-related tags dominate the catalog. Positive conversational evidence is concentrated among several globally popular artists.")
        tags = pd.DataFrame(summary["catalog"]["top_tags"]).head(12).sort_values("tracks")
        sns.barplot(data=tags, y="tag", x="tracks", ax=axes[0], color=GREEN)
        axes[0].set_title("Most common normalized tags")
        axes[0].set_xlabel("Catalog tracks")
        axes[0].set_ylabel("")
        artists = pd.DataFrame(summary["catalog"]["top_positive_artists"]).head(12).sort_values("positive_turns")
        sns.barplot(data=artists, y="artist_name_primary", x="positive_turns", ax=axes[1], color=ORANGE)
        axes[1].set_title("Artists with most positive turns")
        axes[1].set_xlabel("Positive train turns")
        axes[1].set_ylabel("")
        fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.87])
        save_page(pdf, fig, page, "05_catalog_tags_and_artists.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        title(fig, "Popularity and outcome", "Positive outcomes are not explained by higher catalog popularity; their mean and median popularity are slightly lower than negative outcomes.")
        labeled = interactions[interactions["is_labeled"]].copy()
        labeled["outcome"] = np.where(labeled["is_positive"], "Positive", "Negative")
        sns.histplot(data=labeled, x="popularity", hue="outcome", bins=30, stat="density", common_norm=False, element="step", fill=False, ax=axes[0], palette={"Positive": BLUE, "Negative": ORANGE})
        axes[0].set_title("Track popularity distribution")
        axes[0].set_xlabel("Catalog popularity score")
        axes[0].set_ylabel("Density")
        sns.boxplot(data=labeled, x="outcome", y="popularity", order=["Positive", "Negative"], ax=axes[1], palette=[BLUE, ORANGE], hue="outcome", legend=False, showfliers=False)
        axes[1].set_title("Popularity by feedback outcome")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Catalog popularity score")
        axes[1].text(0.02, 0.97, "Positive mean 47.45, median 50\nNegative mean 48.36, median 51", transform=axes[1].transAxes, va="top", fontsize=10, color=GRAY, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
        fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.87])
        save_page(pdf, fig, page, "06_popularity_and_outcome.png")

        page += 1
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1, 1.2]})
        title(fig, "Retrieval evaluation", "Similar-user retrieval outperforms a popularity baseline on all four ranking metrics, but absolute performance remains low.")
        sim = summary["evaluation"]["similar_user"]
        pop = summary["evaluation"]["popularity_baseline"]
        metric_names = ["precision@20", "recall@20", "ndcg@20", "hit_rate@20"]
        metric_labels = ["Precision@20", "Recall@20", "NDCG@20", "Hit Rate@20"]
        eval_frame = pd.DataFrame({"Metric": metric_labels, "Similar user": [sim[m] for m in metric_names], "Popularity": [pop[m] for m in metric_names]}).melt("Metric", var_name="Model", value_name="Score")
        sns.barplot(data=eval_frame, x="Metric", y="Score", hue="Model", ax=axes[0], palette={"Similar user": BLUE, "Popularity": ORANGE})
        axes[0].set_title("Internal temporal validation")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Score")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].tick_params(axis="x", rotation=20)
        label_bars(axes[0], percent=True, decimals=2)
        subgroup = pd.DataFrame(summary["evaluation"]["subgroups"])
        goal = subgroup[subgroup["group_dimension"] == "goal_category"].sort_values("hit_rate@20")
        sns.barplot(data=goal, y="group_value", x="hit_rate@20", ax=axes[1], color=GREEN)
        axes[1].set_title("Hit Rate@20 by discovery goal")
        axes[1].set_xlabel("Hit Rate@20")
        axes[1].set_ylabel("Goal category")
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        label_bars(axes[1], percent=True, horizontal=True)
        fig.tight_layout(rect=[0.03, 0.06, 0.99, 0.87])
        save_page(pdf, fig, page, "07_retrieval_evaluation.png")

        page += 1
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig.text(0.06, 0.93, "Cold start, candidate coverage, and data quality", fontsize=20, fontweight="bold")
        fig.text(0.06, 0.88, "Every test user receives 100 candidates, but embedding availability does not perfectly match declared user splits.", fontsize=10, color=GRAY)
        metrics = [
            ("500 / 500", "test users served"),
            ("396", "similarity eligible"),
            ("104", "popularity fallback"),
            ("0", "test sessions in train history"),
            ("0", "validation sessions in train history"),
            ("0", "missing catalog joins"),
        ]
        for idx, (value, label) in enumerate(metrics):
            col = idx % 3
            row = idx // 3
            x = 0.06 + col * 0.31
            y = 0.73 - row * 0.16
            fig.text(x, y, value, fontsize=22, fontweight="bold", color=BLUE)
            fig.text(x, y - 0.04, label, fontsize=10, color=GRAY)
        findings = [
            "Profiles declare 400 test_warm and 100 test_cold users.",
            "Embedding files provide 371 test_warm vectors and 25 usable vectors in the test_cold file.",
            "All 100 declared cold users and four declared warm users require popularity fallback.",
            "The 16,199 unlabelled rows are the final music turn of each session and are expected.",
            "No duplicate keys, invalid dates, missing catalog joins, test leakage, or validation leakage were found.",
            "Eighty-seven catalog tracks have no normalized tags.",
        ]
        fig.text(0.06, 0.43, "Important findings", fontsize=13, fontweight="bold")
        y = 0.38
        for finding in findings:
            fig.text(0.075, y, "- " + finding, fontsize=10)
            y -= 0.045
        fig.text(0.06, 0.075, "Recommended action: keep the leakage-safe split, report fallback results separately, and stratify evaluation by discovery goal. Treat age 50s and 60+ results as descriptive because their samples are very small.", fontsize=10, color=GRAY)
        save_page(pdf, fig, page, "08_quality_and_cold_start.png")

        page += 1
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        fig.text(0.06, 0.93, "EDA conclusions and next experiments", fontsize=20, fontweight="bold")
        conclusions = [
            ("Collaborative signal exists", "91.5% of users with positive evidence share at least one track with another user, so neighbor-based retrieval is feasible."),
            ("The matrix is extremely sparse", "Median positive history is six tracks per user, and 65.1% of observed positive tracks appear only once."),
            ("Similar users add value", "The method more than doubles Recall@20 and Hit Rate@20 relative to popularity, although absolute scores are still modest."),
            ("Goal category matters", "Goal D has the strongest Hit Rate@20, while goals J, I, and E need focused error analysis."),
            ("Popularity bias is not sufficient", "Positive tracks are slightly less popular on average than negative tracks, so a popularity-only strategy misses useful signal."),
            ("Cold start needs an explicit route", "Use content/profile retrieval or popularity fallback when CF-BPR vectors are unavailable, and report this cohort separately."),
        ]
        y = 0.84
        for heading, body in conclusions:
            fig.text(0.07, y, heading, fontsize=12, fontweight="bold", color=BLUE)
            fig.text(0.07, y - 0.045, "\n".join(textwrap.wrap(body, 105)), fontsize=10, color=GRAY, linespacing=1.4)
            y -= 0.12
        save_page(pdf, fig, page, "09_conclusions.png")

    key_metrics = pd.DataFrame([
        ["sessions", 16199], ["unique_users", 8720], ["catalog_tracks", 47071],
        ["music_turns", 129592], ["labeled_turns", 113393], ["positive_turns", 60069],
        ["positive_rate_labeled", 0.5297], ["user_item_sparsity", 0.99987005],
        ["positive_catalog_coverage", 0.5554], ["users_with_shared_track_fraction", 0.9150],
    ], columns=["metric", "value"])
    key_metrics.to_csv(OUT / "eda_key_metrics.csv", index=False)
    pd.DataFrame(summary["goal_stats"]).to_csv(OUT / "feedback_by_goal.csv", index=False)
    pd.DataFrame(summary["evaluation"]["subgroups"]).to_csv(OUT / "validation_metrics_by_subgroup.csv", index=False)
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = """# TalkPlay static EDA

Contents:

- `talkplay_eda_report.pdf`: ten-page EDA report.
- `figures/`: one high-resolution PNG per report page.
- `eda_key_metrics.csv`: headline dataset metrics.
- `feedback_by_goal.csv`: discovery-goal feedback counts and rates.
- `validation_metrics_by_subgroup.csv`: validation metrics by age, gender, and goal.
- `eda_summary.json`: complete calculated summary used by the report.
- `../scripts/build_eda_summary.py` and `../scripts/create_static_eda.py`:
  reproducible analysis scripts.

The interaction data represents conversational recommendations with next-turn goal-progress feedback. It must not be described as raw listening history.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    zip_path = BASE / "talkplay_static_eda.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
            archive.write(path, arcname=f"talkplay_static_eda/{path.relative_to(OUT).as_posix()}")
    print(f"Created {PDF_PATH}")
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
