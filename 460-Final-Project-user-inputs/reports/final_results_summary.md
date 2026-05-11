# Final Results Summary

## Project Alignment

This build preserves the original proposal by predicting `G3` on the combined (Portuguese + Math) UCI Student Performance dataset with one-hot preprocessing, an 80/20 validation split, 5-fold cross-validation, and a model progression from linear regression to decision trees and MLP. The TA feedback is addressed through interaction features, actionable scenario simulation, cited related work, and rule-based educator intervention plans.

## Best Models

- Best cross-validated model: `lasso` in `with_prior_grades__with_interactions` with RMSE `1.557`.
- Best 80/20 test model: `lasso` in `with_prior_grades__baseline_features` with RMSE `1.478` and pass/fail accuracy `0.876`.
- Selected early-warning model: `lasso` in `early_warning__with_interactions` with RMSE `3.390`.

## Ablation Findings

- `Early warning, baseline features`: best model `mlp`, CV RMSE `3.399`, MAE `2.493`.
- `Early warning, interaction features`: best model `mlp`, CV RMSE `3.366`, MAE `2.455`.
- `Prior grades, baseline features`: best model `lasso`, CV RMSE `1.561`, MAE `0.951`.
- `Prior grades, interaction features`: best model `lasso`, CV RMSE `1.557`, MAE `0.952`.

## Scenario Simulation

- Simulated interventions for `12` at-risk students.
- Average best predicted grade lift: `3.246`.
- Students with at least one scenario reaching the pass threshold: `0`.

## Generated Artifacts

- `reports/metrics.csv`
- `reports/cross_validation_metrics.csv`
- `reports/ablation_summary.csv`
- `reports/feature_importance.csv`
- `reports/scenario_summary.csv`
- `reports/sample_intervention_plans.md`
- `reports/figures/`
