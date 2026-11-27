# Prepared TalkPlay Dataset and EDA

This folder contains a leakage-safe preparation of the public
[TalkPlayData Challenge collection](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge),
the code used for exploratory analysis, and the resulting figures and report.

## Important interpretation boundary

Rows in `conversation_interactions.parquet` represent tracks recommended during
synthetic conversations. A recommendation at music turn `t` is labelled using
the goal-progress assessment in user turn `t+1`. These rows are **conversational
pseudo-interactions**, not verified listening-history events.

The separate public listening-history dataset cannot be joined to challenge
users because it uses unrelated numeric user IDs, while the challenge uses
UUIDs. User similarity therefore uses the official 128-dimensional CF-BPR user
embeddings supplied with the challenge.

## Folder structure

```text
prepared-dataset-eda/
|-- data/                 Prepared Parquet tables and quality summaries
|-- eda/                  PDF report, PNG figures, CSV tables, and JSON summary
|   `-- figures/          Ten high-resolution static EDA images
|-- scripts/              Reproducible EDA scripts
|-- DATA_DICTIONARY.md    Table-level documentation
|-- README.md
`-- requirements.txt
```

## Headline findings

- 16,199 sessions, 8,720 unique users, and 47,071 catalog tracks.
- 129,592 music turns, including 113,393 turns with next-turn feedback.
- 52,550 unique positive user-track pseudo-interactions across 7,790 users.
- The training user-item matrix is 99.987% sparse.
- Positive evidence covers 55.5% of the catalog.
- 91.5% of users with positive evidence share at least one track with another
  user, so neighbor retrieval is feasible despite the sparsity.
- Similar-user retrieval improves Recall@20 from 0.0174 to 0.0374 and Hit
  Rate@20 from 0.0488 to 0.1051 relative to a popularity baseline.
- Every test user receives 100 candidates. Similarity serves 396 users and 104
  require popularity fallback.
- No test sessions or held-out validation sessions enter the training history.

## Reproduce the EDA

From the repository root, create and populate a local virtual environment:

```powershell
py -m venv prepared-dataset-eda\.venv
prepared-dataset-eda\.venv\Scripts\python.exe -m pip install -r prepared-dataset-eda\requirements.txt
```

In VS Code, open `eda.ipynb` and select the `prepared-dataset-eda\.venv`
Python interpreter as the notebook kernel. The environment includes `ipykernel`
for notebook execution.

To regenerate the static EDA outputs, run these commands from this directory
after activating the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
python scripts/build_eda_summary.py
python scripts/create_static_eda.py
```

The first script recalculates `eda/eda_summary.json`. The second regenerates
the PDF report, PNG figures, and CSV summaries.

## Recommended experimental use

Use `internal_train_history.parquet` to generate candidates and
`internal_validation_targets.parquet` only for evaluation. Do not use official
test interactions to select neighbors, build histories, or tune candidate
scores. Compare:

1. Popularity baseline.
2. Similar-user candidates alone.
3. Existing TalkPlay-Tools candidates alone.
4. Combined candidates with intent-aware reranking.

Report Precision@20, Recall@20, NDCG@20, Hit Rate@20, catalog coverage,
popularity, and performance by discovery goal. The dialogue prefixes required
for intent-aware reranking are in `turn_queries.parquet`.

## Sources

- [TalkPlayData Challenge Dataset](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset)
- [Track Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata)
- [User Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata)
- [User Embeddings](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings)
