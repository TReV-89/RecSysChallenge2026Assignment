# Run the TalkPlay evaluation

This guide evaluates the **merged `workI` ReAct recommendation agent**, using the
training evidence in **`prepared-dataset-eda`** and the evaluation scripts on
**`main`**. Evaluation defaults to Groq for recommendations and explanations,
with **LLM judging disabled**. If a Groq turn falls back locally, the evaluator
retries that turn with the configured free OpenRouter model, then retries once
more if OpenRouter itself returns a local fallback.

Start with one complete session (eight turns). That checks the pipeline; it is
too small to estimate overall recommendation quality reliably.

## 1. Open the project and check setup

Run these commands in PowerShell. All subsequent paths are relative to
`agentic_system`.

```powershell
Set-Location 'C:\Users\hp\OneDrive\Desktop\RS\RecSysChallenge\RecSysChallenge2026Assignment\agentic_system'
git status --short --branch
```

The existing `.venv-stable` environment is already configured on this machine.
For a fresh checkout, create it if missing and install the dependencies:

```powershell
if (-not (Test-Path '.venv-stable\Scripts\python.exe')) {
    py -3.14 -m venv .venv-stable
}
.\.venv-stable\Scripts\python.exe -m pip install -e '.[dev]' -c constraints-windows.txt
```

Edit the existing `.env` file; do not overwrite it or commit API keys. These are
the relevant settings (replace the placeholder locally):

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_OUTPUT_METHOD=json_schema
GROQ_REASONING_EFFORT=low
LLM_PROVIDER=groq
```

An OpenRouter key is optional while Groq succeeds, but automatic evaluation
fallback requires `OPEN_ROUTER_API_KEY` or `OPENROUTER_API_KEY`. The fallback is
limited to a model whose name ends in `:free`, and each fallback turn checks the
daily request allowance first. This failover applies only to evaluation; the
interactive API does not switch providers automatically.
`--explanation-provider same` keeps explanations on the active provider.
Always include `--no-judge`: the current CLI otherwise enables its configured
judge by default.

## 2. Prepare the catalog and collaborative data

The full catalog is already built on this machine. On a fresh setup, or if the
runner reports that the full catalog is missing/not ready, build it with:

```powershell
.\.venv-stable\Scripts\python.exe -m music_recommender.ingest
```

This downloads data and builds local indexes, so initial setup can take time.
Do not use `--sample` for evaluation: the evaluator requires the full catalog.

Verify the prepared tables, then import training interactions and published
test-user query vectors:

```powershell
.\.venv-stable\Scripts\python.exe ..\prepared-dataset-eda\scripts\verify_recommender_contract.py
.\.venv-stable\Scripts\python.exe -m music_recommender.cf_data --prepared-dir ../prepared-dataset-eda/data --mode official_test
```

The imported dataset currently contains 53,885 eligible training interaction rows,
52,550 distinct user–track pairs, and 7,790 users. Published query vectors are
available for 396 test users; 104 users have no usable vector and use text retrieval.
Rerun the importer when changing the prepared data or setting up another catalog.

### What data is used?

| Purpose | Source |
|---|---|
| Collaborative neighbor evidence | Positive, labeled **training** rows from `prepared-dataset-eda/data/conversation_interactions.parquet` |
| Track metadata and embeddings | Full local catalog in `data/catalog`, built from the challenge datasets |
| Test-user collaborative query vectors | Published test vectors stored separately from the training neighbor pool |
| Evaluation requests and target tracks | Public test split of `talkpl-ai/TalkPlayData-Challenge-Dataset` |

The evaluator does not simply score the prepared EDA candidate lists. It runs the
agent and scores its actual predictions. Current targets, future turns, hidden
feedback, and test interaction rows are not supplied as retrieval evidence.
Earlier recorded conversation turns are provided as context (*teacher-forced*
evaluation), rather than replacing that context with the agent's own prior output.

## 3. Run a low-cost three-turn pilot without a judge

Choose a **new output folder** for a new experiment. This example avoids the
folders used by earlier runs:

```powershell
$env:GROQ_REASONING_EFFORT = 'low'
$evaluationOutput = 'data/evaluations/my-workI-pilot-01-three-turn'

.\.venv-stable\Scripts\python.exe -u -m music_recommender.evaluate `
  --sessions 1 `
  --turns-per-session 3 `
  --seed 42 `
  --mode groq `
  --explanation-provider same `
  --prepared-dir ../prepared-dataset-eda/data `
  --no-judge `
  --delay 60 `
  --output $evaluationOutput
```

Keep the terminal open. Successful turns print `Completed N/3 turns`.
The runner saves each completed prediction before proceeding to the next turn.
Do not run two processes against the same output directory. Concurrent runs also
share the account's provider quota.
The ReAct tool loop defaults to a 512-token output allowance to stay within
Groq's request limits; explanation generation uses a separate 4096-token budget.
The run manifest records both the Groq primary and OpenRouter fallback model.
This pilot evaluates only the first three turns of one sampled session; its score
is a partial-session check, not a full eight-turn evaluation.

`--delay 60` pauses between complete turns, not between every internal agent call.
The ReAct agent can make several calls per turn, so pacing does not guarantee
that provider rate limits will be avoided. There is also a pause after the final
prediction before scoring. Wait for the scores and `status: complete`.

## 4. Check progress and read the results

In another PowerShell terminal, return to `agentic_system` and set the same output
path before inspecting it:

```powershell
$evaluationOutput = 'data/evaluations/my-workI-pilot-01'
Get-Content "$evaluationOutput/run_status.json"
Get-Content "$evaluationOutput/scores.json"
```

The status file may not exist until the first prediction or a handled failure.
No final `scores.json` is produced until every requested turn is present.

| File | Contents |
|---|---|
| `inputs.json` | Saved test sample, dataset revisions, and full catalog ID list |
| `manifest.json` | Model settings, code hash, input hash, prepared-history file hash, and query-vector hash |
| `checkpoint.jsonl` | Completed predictions, notices, generation mode, and elapsed time per turn |
| `run_status.json` | Completion count and running/stopped/complete status |
| `last_failure.json` | Failed turn and validation/provider notices, when applicable |
| `coverage.json` | Catalog coverage and targets absent from the searchable index |
| `predictions.json` | Final prediction export |
| `scores.json` | Final aggregate metrics and per-turn ranking scores |

An old failure file can remain after a successful resume; check the current status
and scores as well. Artifacts live under ignored `data/` and are not committed.

### Metrics and score interpretation

| Metric | Calculation |
|---|---|
| nDCG@1, @10, @20 | One target track per turn: `1 / log2(rank + 1)` if it appears within the cutoff, otherwise zero; macro-averaged across sessions and turns |
| Catalog diversity | Distinct recommended track IDs across the run / full catalog size |
| Lexical diversity (Distinct-2) | Unique response bigrams / all response bigrams; lowercase whitespace tokens, punctuation retained, no bigrams across responses |

The primary ranking metric is **nDCG@20**. With no judge, the reported weighted
subtotal is:

```text
0.50 × nDCG@20 + 0.10 × catalog_diversity + 0.10 × lexical_diversity
```

Its maximum is **0.70**, not 1.00. The judge and composite fields stay `null`;
the runner does not silently renormalize the weights. This is not a full official
challenge score. Diversity changes with sample size, so compare runs on matching
samples. A single-session pilot has no useful session-bootstrap confidence interval.

## 5. Resume or recompute scores

After a transient rate-limit failure, wait for the provider allowance to recover
and rerun the **same command with the same output path**. Completed turns are
skipped. Keep the same session count, seed, model, settings, source code, and data.
You may increase `--delay` when resuming.

Use a new output folder after changing code, model settings, sample parameters,
or prepared data. The manifest check rejects incompatible resumes so predictions
from different systems are not mixed. Do not edit the manifest to bypass it.

For a completed run, recompute the no-judge scores without regenerating predictions:

```powershell
.\.venv-stable\Scripts\python.exe -m music_recommender.evaluate `
  --sessions 1 --seed 42 --mode groq --no-judge --score-only `
  --turns-per-session 3 `
  --output $evaluationOutput
```

Scoring still requires all requested predictions; it does not drop failed turns.
Use the session count and seed originally used for that output directory.

## 6. Increase the evaluation size

Once the pilot completes, rerun the generation command with a new output folder:

| `--sessions` | Turns with default `--turns-per-session 8` |
|---|---:|
| `1` | 8 |
| `10` | 80 |
| `100` | 800 |
| `0` | Full test split: currently 8,000 |

Set `--turns-per-session 3` to evaluate only the first three turns of each
selected session. This reduces provider calls but is not directly comparable to
full-session metrics.

Seed 42 makes sampling reproducible for the same dataset revision and sample size.
The input snapshot is the definitive record of the sample used. Larger runs can
take many hours or span provider quota resets. A ReAct turn has a variable number
of requests; there is no fixed three-call cost per turn.

## Troubleshooting

- **RateLimitError:** the evaluator tries the configured free OpenRouter model
  after a Groq local fallback. If that model's allowance is low, the run stops
  with its completed checkpoint; resume the same command after the allowance resets.
  A larger between-turn delay may help with per-minute limits but cannot remove daily caps.
- **BadRequestError / invalid explanation:** inspect `last_failure.json`. Tag
  corrections must use literal supplied tags and still pass per-track validation.
  Explanation generation has its own prompt, separate from ReAct tool selection.
  Do not enable `--allow-fallbacks` just to get a passing run; that is a different,
  explicitly mixed-output experiment.
- **“configuration/code changed”:** choose a fresh output folder; preserve the old
  run as a separate experiment.
- **Query vectors not ready:** run `music_recommender.cf_data` from step 2 without
  `--skip-test-vectors`.
- **Missing prepared Parquet files:** check `--prepared-dir`; it should point to
  the merged repository's `prepared-dataset-eda/data` directory.
- **Chroma/index errors:** stop concurrent catalog operations and inspect the
  catalog error. Do not delete the catalog or old results as a first recovery step.
- **Missing packages:** rerun the dependency installation in step 1 using the
  project's `.venv-stable` interpreter.

To inspect all supported arguments:

```powershell
.\.venv-stable\Scripts\python.exe -m music_recommender.evaluate --help
```
