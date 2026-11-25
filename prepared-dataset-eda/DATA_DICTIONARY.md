# Data Dictionary

## Core tables

### `sessions.parquet`

One row per conversational session. It contains the session and user IDs,
source split, date, demographic fields, preferred language and musical culture,
and the conversation discovery goal.

### `conversation_interactions.parquet`

One row per music recommendation. Key fields:

- `music_turn`: recommendation turn number.
- `feedback_turn`: following user turn used for the label.
- `track_id`: recommended track UUID.
- `goal_progress_label`: `MOVES_TOWARD_GOAL`,
  `DOES_NOT_MOVE_TOWARD_GOAL`, or null for the final turn.
- `is_positive`, `is_negative`, `is_labeled`: convenient Boolean labels.
- `interaction_semantics`: explicit reminder that the row is a conversational
  recommendation with next-turn feedback.

### `turn_queries.parquet`

One row per recommendation point. It contains the current user request and the
dialogue prefix available before the recommendation. Join user-level candidates
to this table before intent-aware reranking.

### `track_metadata_clean.parquet`

The 47,071-track catalog with primary track, artist, and album display values;
release information; popularity; original metadata arrays; and normalized,
case-folded, within-track-deduplicated tags.

## Leakage-safe development split

### `internal_train_history.parquet`

Positive training pseudo-interactions available before each eligible user's
latest held-out training session.

### `internal_validation_targets.parquet`

Positive targets from the held-out latest training sessions. These rows must
not be used for neighbor selection or candidate generation.

### `internal_validation_candidates.parquet`

Top-100 similar-user candidates for internal temporal validation. Tracks already
seen in the user's development history are removed before ranking.

### `internal_validation_metrics_by_user.parquet`

Per-user Precision@20, Recall@20, NDCG@20, and Hit Rate@20.

## Similar users and official-test candidates

### `user_neighbors.parquet`

Top-30 training neighbors for each training user and each test user with a
usable embedding. Similarity is cosine similarity on the official CF-BPR user
vectors. The neighbor pool always contains training users only.

### `official_test_candidates.parquet`

Exactly 100 candidates for each of the 500 test users. Candidate partitions
identify similarity-based retrieval and popularity fallback.

## Quality and subgroup summaries

- `data_quality_report.json`: source counts, feedback semantics, sparsity,
  overlap, evaluation metrics, and leakage checks.
- `subgroup_metrics.csv`: Precision@20, Recall@20, NDCG@20, and Hit Rate@20 by
  age group, gender, and discovery goal.

## Known limitations

- Conversation recommendations are not raw historical listens.
- Histories are short: the median user has six unique positive tracks.
- The matrix is extremely sparse and long-tailed.
- Small older-age subgroups should be treated as descriptive only.
- Declared session splits and embedding-file splits differ for 29 test users;
  candidate routing is based on actual vector availability.
