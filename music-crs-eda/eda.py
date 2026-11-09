import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = r"C:\Users\aaron\Downloads\music-crs-eda"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("--- Step 1: Loading Datasets ---")
df_train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
df_test = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
df_tracks = pd.read_parquet(os.path.join(DATA_DIR, "tracks_metadata.parquet"))
df_users = pd.read_parquet(os.path.join(DATA_DIR, "users_metadata.parquet"))

print(f"Train sessions: {len(df_train):,}")
print(f"Test sessions:  {len(df_test):,}")
print(f"Tracks catalog: {len(df_tracks):,}")
print(f"Users catalog:  {len(df_users):,}")

# Combine train and test for comprehensive session-level statistics (with split flag)
df_train['split'] = 'train'
df_test['split'] = 'test'
df_all_sessions = pd.concat([df_train, df_test], ignore_index=True)

print("\n--- Step 2: Unpacking Session-Level Metadata ---")
# Unpack user_profile
user_profiles = pd.json_normalize(df_all_sessions['user_profile'])
user_profiles.columns = [f"up_{c}" for c in user_profiles.columns]

# Unpack conversation_goal
goals = pd.json_normalize(df_all_sessions['conversation_goal'])
goals.columns = [f"goal_{c}" for c in goals.columns]

# Attach to sessions
df_sess_flat = pd.concat([df_all_sessions[['session_id', 'user_id', 'session_date', 'split']], user_profiles, goals], axis=1)

print("Sample flattened record columns:")
print(df_sess_flat.columns.tolist())

# Basic counts
print("\nUnique Users in Sessions:", df_sess_flat['user_id'].nunique())
print("Sessions per User Distribution:")
sess_per_user = df_sess_flat.groupby('user_id')['session_id'].count()
print(sess_per_user.describe())

# Conversation Goal breakdown
print("\nGoal Categories:")
print(df_sess_flat['goal_category'].value_counts(dropna=False))
print("\nGoal Specificity:")
print(df_sess_flat['goal_specificity'].value_counts(dropna=False))

# Demographic distributions
print("\nAge Group Breakdown:")
print(df_sess_flat['up_age_group'].value_counts(dropna=False))
print("\nTop 10 Countries:")
print(df_sess_flat['up_country_name'].value_counts(dropna=False).head(10))
print("\nGender Breakdown:")
print(df_sess_flat['up_gender'].value_counts(dropna=False))
print("\nTop Musical Cultures:")
print(df_sess_flat['up_preferred_musical_culture'].value_counts(dropna=False).head(10))
print("\nTop Preferred Languages:")
print(df_sess_flat['up_preferred_language'].value_counts(dropna=False).head(10))

print("\n--- Step 3: Analyzing Turn-Level Conversations & Assessments ---")
# Conversation statistics
turn_counts = []
user_utterance_lengths = []
agent_utterance_lengths = []
thought_lengths = []
has_thought_count = 0
total_agent_turns = 0
total_user_turns = 0

all_progress_assessments = []

for idx, row in df_all_sessions.iterrows():
    convs = row['conversations']
    if convs is not None and len(convs) > 0:
        turn_counts.append(len(convs))
        for turn in convs:
            role = turn.get('role')
            content = turn.get('content') or ''
            words = len(content.split())
            if role == 'user':
                total_user_turns += 1
                user_utterance_lengths.append(words)
            elif role in ['assistant', 'system', 'agent']:
                total_agent_turns += 1
                agent_utterance_lengths.append(words)
                th = turn.get('thought')
                if th:
                    has_thought_count += 1
                    thought_lengths.append(len(th.split()))
    
    assessments = row['goal_progress_assessments']
    if assessments is not None and len(assessments) > 0:
        for a in assessments:
            all_progress_assessments.append({
                'session_id': row['session_id'],
                'split': row['split'],
                'turn_number': a.get('turn_number'),
                'assessment': a.get('goal_progress_assessment')
            })

turn_series = pd.Series(turn_counts)
print("\nTurns per Conversation Summary:")
print(turn_series.describe())

print(f"\nTotal User Turns: {total_user_turns:,}, Mean word count: {np.mean(user_utterance_lengths):.1f} (std: {np.std(user_utterance_lengths):.1f})")
print(f"Total Agent Turns: {total_agent_turns:,}, Mean word count: {np.mean(agent_utterance_lengths):.1f} (std: {np.std(agent_utterance_lengths):.1f})")
print(f"Agent Turns with Thoughts: {has_thought_count:,} / {total_agent_turns:,} ({has_thought_count/max(1, total_agent_turns)*100:.1f}%)")
if thought_lengths:
    print(f"Agent Thought Mean word count: {np.mean(thought_lengths):.1f} (std: {np.std(thought_lengths):.1f})")

df_assess = pd.DataFrame(all_progress_assessments)
if not df_assess.empty:
    print("\nGoal Progress Assessments Breakdown:")
    print(df_assess['assessment'].value_counts(dropna=False))
    print("\nAssessments by Turn Number (Top 5 turns):")
    print(pd.crosstab(df_assess['turn_number'], df_assess['assessment']).head(6))

print("\n--- Step 4: Analyzing Tracks Metadata ---")
print("Tracks DataFrame columns:", df_tracks.columns.tolist())
print(df_tracks.info())
print("\nPopularity describe:")
print(df_tracks['popularity'].describe())
print("\nDuration (seconds) describe:")
print((df_tracks['duration'] / 1000).describe())

# Tags analysis
all_tags = []
for tags in df_tracks['tag_list'].dropna():
    if hasattr(tags, '__iter__') and not isinstance(tags, str):
        all_tags.extend(tags)
    elif isinstance(tags, str):
        all_tags.append(tags)

tag_counts = Counter(all_tags)
print(f"\nTotal unique tags: {len(tag_counts):,}")
print("Top 20 most frequent tags:")
for tag, count in tag_counts.most_common(20):
    print(f"  {tag}: {count:,}")

# Artists analysis
all_artists = []
for artists in df_tracks['artist_name'].dropna():
    if hasattr(artists, '__iter__') and not isinstance(artists, str):
        all_artists.extend(artists)
    elif isinstance(artists, str):
        all_artists.append(artists)
artist_counts = Counter(all_artists)
print(f"\nTotal unique artist occurrences: {len(artist_counts):,}")
print("Top 10 most cataloged artists:")
for artist, count in artist_counts.most_common(10):
    print(f"  {artist}: {count:,}")

print("\n--- Step 5: Generating Visualizations ---")
# Figure 1: Distribution of Conversation Turns & Goal Categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(turn_series, bins=range(min(turn_counts), max(turn_counts)+2), kde=False, color='#2b5c8f', ax=axes[0], discrete=True)
axes[0].set_title("Distribution of Conversation Turns per Session", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Number of Turns")
axes[0].set_ylabel("Number of Sessions")
axes[0].axvline(turn_series.median(), color='#d95f02', linestyle='--', label=f'Median: {turn_series.median():.0f}')
axes[0].axvline(turn_series.mean(), color='#7570b3', linestyle=':', label=f'Mean: {turn_series.mean():.1f}')
axes[0].legend()

# Goal category bar chart
goal_cat_counts = df_sess_flat['goal_category'].value_counts()
sns.barplot(x=goal_cat_counts.values, y=goal_cat_counts.index, palette='Blues_r', ax=axes[1])
axes[1].set_title("Distribution of Music Discovery Goal Categories", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Number of Sessions")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_turns_and_goals.png"), dpi=200)
plt.close()

# Figure 2: User Demographics (Age Group, Gender, Top Countries, Musical Culture)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age groups
age_order = ['18-24', '25-34', '35-44', '45-54', '55+']
age_counts = df_sess_flat['up_age_group'].value_counts()
sns.barplot(x=age_counts.index, y=age_counts.values, palette='crest', ax=axes[0, 0])
axes[0, 0].set_title("Listener Age Group Distribution", fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel("Count")

# Gender
gender_counts = df_sess_flat['up_gender'].value_counts()
axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])
axes[0, 1].set_title("Listener Gender Distribution", fontsize=11, fontweight='bold')

# Top 8 Countries
top_countries = df_sess_flat['up_country_name'].value_counts().head(8)
sns.barplot(x=top_countries.values, y=top_countries.index, palette='mako', ax=axes[1, 0])
axes[1, 0].set_title("Top 8 Listener Countries", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel("Count")

# Top 8 Musical Cultures
top_cultures = df_sess_flat['up_preferred_musical_culture'].value_counts().head(8)
sns.barplot(x=top_cultures.values, y=top_cultures.index, palette='rocket', ax=axes[1, 1])
axes[1, 1].set_title("Top 8 Preferred Musical Cultures", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel("Count")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_user_demographics.png"), dpi=200)
plt.close()

# Figure 3: Track Catalog Analysis (Popularity, Duration, Top Tags)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Popularity
sns.histplot(df_tracks['popularity'], bins=30, kde=True, color='#2ca02c', ax=axes[0])
axes[0].set_title("Track Popularity Score Distribution", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Popularity (0-100)")
axes[0].set_ylabel("Number of Tracks")

# Duration in minutes
dur_mins = df_tracks['duration'] / (1000 * 60)
sns.histplot(dur_mins[dur_mins < 15], bins=30, kde=True, color='#1f77b4', ax=axes[1])
axes[1].set_title("Track Duration (Minutes, <15m)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Duration (Minutes)")
axes[1].set_ylabel("Number of Tracks")

# Top 12 Tags
top_12_tags = dict(tag_counts.most_common(12))
sns.barplot(x=list(top_12_tags.values()), y=list(top_12_tags.keys()), palette='viridis', ax=axes[2])
axes[2].set_title("Top 12 Most Common Track Tags / Genres", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Track Count")

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_track_catalog.png"), dpi=200)
plt.close()

# Figure 4: Goal Progress Assessment Progression
if not df_assess.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df_assess['turn_number'], df_assess['assessment'], normalize='index') * 100
    ct_top = ct.loc[ct.index <= 8] # First 8 turns
    ct_top.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax)
    ax.set_title("Goal Progress Assessment Breakdown Across Conversation Turns (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Conversation Turn Number")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Assessment Status", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_goal_progression.png"), dpi=200)
    plt.close()

# Save detailed summary stats as JSON
summary_stats = {
    "total_sessions": len(df_all_sessions),
    "train_sessions": len(df_train),
    "test_sessions": len(df_test),
    "unique_users_sessions": int(df_sess_flat['user_id'].nunique()),
    "total_catalog_tracks": len(df_tracks),
    "total_catalog_users": len(df_users),
    "turns_summary": {
        "min": int(turn_series.min()),
        "max": int(turn_series.max()),
        "mean": float(turn_series.mean()),
        "median": float(turn_series.median()),
        "std": float(turn_series.std())
    },
    "goal_categories": df_sess_flat['goal_category'].value_counts().to_dict(),
    "goal_specificity": df_sess_flat['goal_specificity'].value_counts().to_dict(),
    "age_groups": df_sess_flat['up_age_group'].value_counts().to_dict(),
    "top_countries": df_sess_flat['up_country_name'].value_counts().head(10).to_dict(),
    "top_musical_cultures": df_sess_flat['up_preferred_musical_culture'].value_counts().head(10).to_dict(),
    "top_tags": dict(tag_counts.most_common(20)),
    "track_popularity": {
        "mean": float(df_tracks['popularity'].mean()),
        "median": float(df_tracks['popularity'].median()),
        "std": float(df_tracks['popularity'].std()),
        "min": float(df_tracks['popularity'].min()),
        "max": float(df_tracks['popularity'].max())
    },
    "user_word_count_mean": float(np.mean(user_utterance_lengths)),
    "agent_word_count_mean": float(np.mean(agent_utterance_lengths)),
    "agent_thought_percentage": float(has_thought_count / max(1, total_agent_turns) * 100),
    "progress_assessments": df_assess['assessment'].value_counts().to_dict() if not df_assess.empty else {}
}

with open(os.path.join(BASE_DIR, "summary_stats.json"), "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2)

print("\n--- EDA Script Completed Successfully! ---")
print(f"Summary JSON written to: {os.path.join(BASE_DIR, 'summary_stats.json')}")
print(f"Figures written to: {FIG_DIR}")

