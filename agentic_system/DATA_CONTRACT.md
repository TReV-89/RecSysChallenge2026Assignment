# Prepared data contract for recommendation

Catalog ingestion loads **training** CF-BPR user vectors. Run
`python -m music_recommender.cf_data` afterward to add eligible prepared
interactions and published official-test query vectors. Query vectors remain
separate from the training neighbor pool. `user_neighbors.parquet` is an EDA
artifact capped at 30 neighbors; the running path scores the full available
training-vector pool with `find_cf_neighbors` instead of using that cap.

| Retrieval setting | Allowed neighbor evidence | Never use as input |
| --- | --- | --- |
| Internal temporal validation | Positive, labeled rows in `internal_train_history.parquet` | `internal_validation_targets.parquet` |
| Official test | Positive, labeled rows with `source_split == "train"` in `conversation_interactions.parquet` | Test interaction rows or evaluation targets |
| New live listener | Only previously observed, eligible interactions | Future conversation turns or hidden feedback |

Join neighbor evidence on `user_id`, then candidate metadata and vectors on
`track_id`. For a validation user, the history must precede that user's held-out
session. Deduplicate repeated `(user_id, track_id)` evidence before scoring a
neighbor's support for a track. Exclude the query user's own tracks when the
relevant past history is available.

These interaction rows represent **tracks recommended in synthetic conversations
with next-turn goal-progress feedback**. A positive label does not establish
that the listener played, liked, or rated the track. Explanations and API fields
must preserve this distinction.

`turn_queries.parquet` supplies the current request and prior dialogue for
intent-aware ranking; it is not neighbor evidence. `track_metadata_clean.parquet`
provides catalog display fields. The published user embedding split and the
session split do not always agree, so a user-to-user or user-to-item route must
check for an actual compatible CF-BPR vector. `cf_data.py` imports the published
`test_warm` and `test_cold` embedding splits. Users without a usable vector use
text retrieval; new live users do not acquire a CF-BPR vector from profile text.

Run `prepared-dataset-eda/scripts/verify_recommender_contract.py` with the
prepared-data environment to check the local files and leakage boundaries.
