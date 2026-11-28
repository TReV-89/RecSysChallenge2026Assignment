# Profile-enhanced CF retrieval: evaluation handoff

## What changed

- The original published 128-dimensional CF-BPR user and track factors remain in
  the catalog. An offline adapter adds a small correction based on
  `preferred_musical_culture` to user factors. It was fitted from eligible
  positive feedback in the **training split only**. These labels describe
  synthetic next-turn goal progress, not confirmed listens or likes.
- User-to-user retrieval compares the enhanced query factor with the enhanced
  training-user pool. User-to-item retrieval compares the enhanced query factor
  with the unchanged CF-BPR track factors. The selected correction strengths are
  1.0 and 0.5, respectively. If the query has no usable culture or no CF-BPR
  factor, the existing fallback behavior applies.
- The full-catalog API and official-test evaluation now default to the
  official training-split adapter and `population_centered` similarity.
  `CF_NEIGHBOR_SIMILARITY=cosine` selects ordinary cosine with the same adapter.
  Population centering subtracts the mean enhanced training-user
  factor from both query and neighbor before cosine. It is **not** the classical
  rating-based adjusted-cosine formula.

## Retrieval-only comparison already completed

The two runs used the same 100 public-test sessions (800 turns), seed 42,
input snapshot, training feedback, official-test adapter, retrieval code and
`user_to_user` source. Only `CF_NEIGHBOR_SIMILARITY` changed.

| Neighbor metric | nDCG@10 | nDCG@20 | Catalog diversity |
| --- | ---: | ---: | ---: |
| Cosine | 0.001381 | 0.002683 | 0.003250 |
| Population-centered cosine | 0.002342 | 0.002983 | 0.004079 |

Centered minus cosine nDCG@20 was **+0.000300**. The paired session-bootstrap
95% interval was **[-0.001436, +0.001940]**, so the 100-session retrieval run
does not establish a reliable winner. Rankings changed on 576 of 800 turns,
but their session-level nDCG@20 was equal on 91 of 100 sessions. This is a
user-to-user retrieval-only comparison; it does not measure agent tool choice,
fusion with text/user-to-item, explanations, or the full pipeline.
On this 100-session sample, all requests had a recognized culture field, but
only 73 users had a published CF-BPR query factor. The adapter changed the
query factor for all 73; the other 27 have no user-to-user or user-to-item CF
route and rely on available text/item routes.

The complete run outputs are in
`data/evaluations/retrieval-profile-cosine-u2u-100/` and
`data/evaluations/retrieval-profile-centered-u2u-100/`. The paired calculation
is in `data/evaluations/retrieval-profile-cosine-vs-centered-u2u-100.json`.
These evaluation directories are ignored by Git, so share their scores or this
summary if a colleague needs the numbers.

## Full-pipeline agent evaluation

For one exploratory full-agent run, use the enhanced adapter with centered
cosine. Compare it with an existing baseline **only when the sessions, seed,
provider mode and other evaluation flags match**. One such comparison tests the
combined change (adapter plus centered neighbors); it cannot attribute a gain
specifically to centering. Testing centering itself in the full agent requires
a second run with the same adapter and `CF_NEIGHBOR_SIMILARITY=cosine`.

From `agentic_system` in PowerShell, these settings make the default explicit:

```powershell
$env:CF_HISTORY_MODE = "official_test"
$env:CF_PROFILE_ADAPTER_ENABLED = "1"
$env:CF_NEIGHBOR_SIMILARITY = "population_centered"
```

Then run the colleague's existing `music_recommender.evaluate` command with a
**new output directory** and the same sample/seed/provider settings as the
baseline. Use the existing 100-session input snapshot if exact input revisions
must match. For the matched cosine agent run, change only
`CF_NEIGHBOR_SIMILARITY` to `cosine` and use another new output directory.
To reproduce the original-factor baseline, set `CF_PROFILE_ADAPTER_ENABLED=0`;
that also restores ordinary cosine. Set `CF_HISTORY_MODE=official_test` before
the colleague's official-test run so the default artifact matches that scope.
Compare nDCG@20, catalog diversity, which tools were called, and the number of
turns completed. Judge or explanation metrics should only be compared when
their provider and settings match too.

The official adapter artifacts are
`../prepared-dataset-eda/eda/culture_adapter_official_weights.npz` (SHA-256
`93B5A8362ADB530A91E4F061812454401310EEF698168BA05CDD78D20F495F87`)
and `../prepared-dataset-eda/eda/culture_adapter_official_users.npz` (SHA-256
`49407CDCD99BAC2200EE89D3978C3ECDF70268029D9802E7B6A4F93EA1DA508B`).
Both files must be included in the push. The separate `culture_adapter_final_*`
artifacts were fitted for internal temporal validation and should not be used
for this official-test pipeline run.
