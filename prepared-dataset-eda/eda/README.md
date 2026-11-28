# TalkPlay static EDA

Contents:

- `talkplay_eda_report.pdf`: ten-page EDA report.
- `figures/`: one high-resolution PNG per report page.
- `eda_key_metrics.csv`: headline dataset metrics.
- `feedback_by_goal.csv`: discovery-goal feedback counts and rates.
- `validation_metrics_by_subgroup.csv`: validation metrics by age, gender, and goal.
- `profile_neighbor_experiment.json`: a held-out comparison of the saved top-30
  BPR neighbors against BPR plus musical-culture similarity. Reproduce it with
  `python scripts/evaluate_profile_similarity.py` from `prepared-dataset-eda`.
  Language is excluded because it is constant in the prepared sessions.
- `culture_adapter_experiment.json`: a full-pool, held-out comparison of
  unchanged CF-BPR factors against a learned musical-culture correction for
  user-to-user and user-to-item retrieval. The fitting users exclude the
  validation cohort; the published track factors stay fixed. It includes a
  shuffled-culture control and paired bootstrap intervals. Reproduce it with
  `../agentic_system/.venv-stable/Scripts/python.exe scripts/evaluate_culture_adapter.py`
  from `prepared-dataset-eda`.
- `culture_adapter_weights.npz`: the experimental token weights and selected
  correction strengths. This is an offline artifact; the application does not
  load it yet.
- `culture_adapter_final_fit.json`, `culture_adapter_final_weights.npz`, and
  `culture_adapter_final_users.npz`: final offline refit using all eligible
  pre-holdout training feedback, with strengths chosen by the earlier
  experiment. The user-vector archive contains separate user-to-user and
  user-to-item vectors for indexed training users. Reproduce these files with
  `../agentic_system/.venv-stable/Scripts/python.exe scripts/train_culture_adapter.py`
  from `prepared-dataset-eda`. The final fit is distinct from the earlier
  held-out model; its effect on the full agent has not been measured.
- `centered_neighbor_experiment.json`: comparison of cosine and training-pool
  mean-centered cosine on the enhanced user vectors, using the earlier adapter
  that excluded validation users from fitting. Reproduce it with
  `../agentic_system/.venv-stable/Scripts/python.exe scripts/evaluate_centered_neighbors.py`.
  The small evaluation gain has a paired interval spanning zero.
- `culture_adapter_official_fit.json`, `culture_adapter_official_weights.npz`,
  and `culture_adapter_official_users.npz`: separate refit using only the full
  official training split for colleagues' official-test pipeline evaluation.
  Reproduce it with `../agentic_system/.venv-stable/Scripts/python.exe
  scripts/train_culture_adapter.py --scope official_test`.
- `eda_summary.json`: complete calculated summary used by the report.
- `../scripts/build_eda_summary.py` and `../scripts/create_static_eda.py`:
  reproducible analysis scripts.

The interaction data represents conversational recommendations with next-turn goal-progress feedback. It must not be described as raw listening history.
