# TalkPlay static EDA

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
