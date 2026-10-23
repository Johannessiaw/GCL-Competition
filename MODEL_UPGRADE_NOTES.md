# Model upgrade notes

## Changes applied to main (v2)

1. Expanded feature engineering: loan-duration proxy (credit/annuity), income-per-child,
   children/family ratio, region-rating x population, car-ownership indicators
   (including car-age vs driver-age), credit-bureau enquiry totals and recency-weighted
   totals, document/contact-flag sums, registration/ID-publish and phone-change recency
   features, social-circle default ratios, EXT_SOURCE std/count, and age buckets.
2. CatBoost and LightGBM retuned (CatBoost: depth 7, lr 0.035, early stopping with
   od_wait=300; LightGBM: num_leaves 48, lr 0.03, min_child_samples 70, early
   stopping 200) with per-fold best-iteration reporting.
3. Multi-seed averaging (42, 2026) per model per fold to stabilise test predictions.
4. Blend weight is now grid-searched on pooled out-of-fold AUC instead of a fixed
   0.55/0.45 split.
5. Prediction sanity checks: fails on (near-)constant test probabilities (the classic
   cause of a ~0.50 leaderboard AUC) and warns on test/OOF distribution drift.
6. OOF predictions are saved to output/oof_predictions.csv for offline analysis.
7. Fold-specific LightGBM categorical maps are now also pandas>=4 safe (unseen
   categories are mapped to NaN explicitly before Categorical construction).
8. Optional env knobs: GCL_TRAIN_PATH (train file override), GCL_CAT_ITERATIONS,
   GCL_LGB_N_ESTIMATORS (smoke-test budgets).

## Validation status

- End-to-end smoke test passed on a synthetic train file (tools/make_synthetic_train.py,
  built from test.csv rows with an injected weak signal): pipeline completed all 5 folds,
  2 seeds, both models, wrote a valid 61,500-row submission, and scored blended OOF
  AUC 0.914 on the synthetic signal (proving the pipeline learns; random labels would
  score ~0.50). This validates code correctness only - it says nothing about real-data
  performance.
- input/train.csv is still absent, so real-data training/CV cannot be rerun here.
  The historical v1 blended AUC of 0.756483 remains the best real-data reference.
- No external data was used at any point (competition rules); improvements come from
  feature engineering and tuning on the official application table only.

## Next steps

1. Put the official train.csv in input/ locally; do not commit course data.
2. Install requirements.txt and run `python final_submission.py` (full budget:
   ~5k CatBoost iters, ~4k LightGBM trees with early stopping; on 2 cores expect
   roughly 1-3 hours. Set SEEDS=(42,) for a faster run).
3. Review fold AUCs and the pooled OOF AUC; compare with the v1 baseline on the same
   folds/seed before changing anything else.
4. Check output/submission.csv locally (row count, ID order, non-constant values)
   and upload the intended file - the last submission counts.

