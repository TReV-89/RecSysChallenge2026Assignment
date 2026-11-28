# Retrieval-only ablation: does CF-BPR retrieval help?

This isolates retrieval+ranking quality from the agent's LLM tool-selection
behavior, to answer a specific question: **does adding our CF-BPR
user-to-user / user-to-item retrieval improve recommendation quality over
text retrieval alone, or not?**

## Method

A new script, `music_recommender.evaluate_retrieval`, replaces the agent's
LLM-driven tool selection with an explicit, fixed choice of retrieval
source(s) for every turn, then runs the result through the exact same
unmodified fusion/ranking/scoring pipeline the full agent uses
(`merge_results` → `rank_candidates` → the challenge's nDCG@k metric). No LLM
is called anywhere in this path — no tool-calling, no explanations — so
`predicted_response` is empty and `lexical_diversity` is `0.0` by
construction (not a real result; just nothing to measure). Everything else
(protocol, teacher-forced history, catalog, target tracks) is identical to
the full-agent evaluation.

Three retrieval sources exist in the system:
- **`text`** — Qwen embedding similarity against the listener's message.
- **`user_to_user`** — CF-BPR neighbor search: pools tracks from the most
  similar other listeners' positive feedback history.
- **`user_to_item`** — direct inner-product ranking of the catalog against
  the listener's own trained CF-BPR factor.

(`item_to_item` is excluded from this ablation — it needs a reference track
the agent would normally pick mid-conversation, with no equivalent in a
single blind retrieval call.)

Protocol: official public test split of `talkpl-ai/TalkPlayData-Challenge-Dataset`,
teacher-forced history, seed 42, **100 sessions / 800 turns**, full
47,071-track catalog.

## Results

| Source(s) | nDCG@1 | nDCG@10 | nDCG@20 | 95% CI (nDCG@20, session bootstrap) | Catalog diversity | Unique tracks recommended |
|---|---:|---:|---:|---:|---:|---:|
| `text` only | 0.0113 | 0.0216 | **0.0235** | [0.0145, 0.0332] | 0.1255 | 5,909 |
| `user_to_item` only | 0.0013 | 0.0039 | **0.0045** | [0.0008, 0.0094] | 0.0208 | 980 |
| `user_to_user` only | 0.0000 | 0.0018 | **0.0028** | [0.0006, 0.0052] | 0.0047 | 221 |
| `text + user_to_user + user_to_item` (all three) | 0.0088 | 0.0189 | **0.0209** | [0.0127, 0.0295] | 0.0893 | 4,202 |

For external reference, the challenge's own evaluator repo publishes these
baselines (measured on the **devset**, not the test split used above — not a
perfectly matched comparison, included for rough context only):

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity |
|---|---:|---:|---:|---:|
| Random | 0.0000 | 0.0001 | 0.0001 | 0.9652 |
| Popularity | 0.0005 | 0.0018 | 0.0024 | 0.0004 |
| LLaMA-1B + BM25 (strongest official baseline) | 0.0098 | 0.0627 | 0.0815 | 0.3795 |

## Findings

1. **Text retrieval alone is carrying essentially all of the real signal.**
   It beats every CF-only variant by 5–8x on nDCG@20, and recommends from a
   far larger, more diverse slice of the catalog (5,909 vs. 980/221 unique
   tracks across 800 turns).
2. **Combining CF with text does not improve on text alone — if anything, it's
   slightly worse** (nDCG@20 0.0209 vs. 0.0235), though the confidence
   intervals overlap enough that this isn't a statistically significant
   regression. Catalog diversity also drops (0.1255 → 0.0893): the narrower
   CF candidate pools pull the fused ranking toward a smaller, less varied
   set whenever a listener has a CF vector.
3. **Between the two CF mechanisms, direct factor matching (`user_to_item`)
   is meaningfully stronger than neighbor-pooled evidence (`user_to_user`)**
   — about 60% higher nDCG@20, and a much wider recommended-track footprint
   (980 vs. 221 unique tracks). Neighbor pooling depends on finding similar
   listeners *and* overlap between their history and the catalog; direct
   factor matching has fewer failure points.
4. **Neither CF channel clears the official Popularity baseline by much.**
   `user_to_item` is roughly 2x Popularity; `user_to_user` is barely above
   it. Text retrieval alone is the only source here that meaningfully beats
   Popularity.
5. **Against the strongest official baseline (LLaMA-1B + BM25), our best
   retrieval-only config (text-only) is ~3.5x behind** (0.0235 vs. 0.0815
   nDCG@20) — noting the devset/test-split caveat above.

**Net takeaway:** at this sample size, the CF-BPR extensions aren't earning
their complexity at the retrieval layer — they don't add signal on top of
text retrieval, and in isolation they're weak relative to text and only
marginally above a trivial popularity baseline. This doesn't necessarily mean
the full conversational agent is unaffected the same way (the LLM may
down-weight CF evidence in ranking/explanation in ways this retrieval-only
test can't see), but the retrieval-level question has a fairly clear answer
right now: CF is not currently pulling its weight.

## Caveats

- 800 turns gives tighter confidence intervals than earlier small pilots, but
  CF-only variants still have wide relative uncertainty (e.g. `user_to_item`'s
  CI spans roughly 2–10x its own point estimate).
- The official baseline table is measured on the **devset**; this project's
  evaluation runs on the **test** split. Same dataset family, different
  partition — informative, not a perfect comparison.
- The original TalkPlay-Tools paper (arXiv:2510.01698) is not a usable
  baseline here: it reports **Hit@K**, not nDCG@K, on its own synthetic
  benchmark, not the official challenge test split. Different metric,
  different data — not directly comparable to any number in this document.
- `lexical_diversity` is `0.0` in every row here by construction (no LLM, no
  generated text) — not a measured result.

## Reproducing this

```powershell
.\.venv-stable\Scripts\python.exe -m music_recommender.evaluate_retrieval `
  --sources text `
  --sessions 100 --turns-per-session 8 --seed 42 `
  --prepared-dir ../prepared-dataset-eda/data `
  --output data/evaluations/retrieval-text-only-100
```

Swap `--sources` for `user_to_user`, `user_to_item`, or any combination
(`text user_to_user user_to_item`, etc.) and use a fresh `--output` folder per
configuration. No LLM/API key needed; each 800-turn run takes roughly
15–20 minutes locally.

## Suggested next step

Isolate which CF channel is responsible for the "combining hurts" effect:
run `text + user_to_item` (dropping `user_to_user` entirely) and compare
against `text`-only and the full three-source run above.
