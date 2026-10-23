# Model upgrade notes

## Changes applied to main (v3)

1. Leakage-safe categorical encodings: frequency encoding plus smoothed target
   encoding computed out-of-fold (each row is encoded with target statistics from
   rows outside its own encoding fold; test rows use full-train statistics with
   the training prior as fallback). Applied to categorical features with 3-100
   unique values.
2. Third model in the blend: XGBoost (hist tree method, categorical-aware via
   the same fold-safe category mappings, early stopping on validation AUC).
   Auto-skipped if xgboost is not installed; disable with GCL_USE_XGB=0.
3. Blend weights are now grid-searched over ALL model OOF predictions (simplex
   grid, 0.05 step) instead of a single CatBoost-vs-LightGBM weight; the test
   submission is a rank-normalised weighted blend (standard Kaggle trick that is
   robust to probability miscalibration between models).
4. New env knobs: GCL_XGB_N_ESTIMATORS, GCL_XGB_EARLY_STOP, GCL_USE_XGB,
   GCL_TE_SMOOTHING (target-encoding smoothing strength, default 10).
5. requirements.txt now includes scipy and xgboost.

## References (public techniques only - no external data used)

- LightGBM parameter reference (num_leaves, min_child_samples, subsampling,
  regularisation, early stopping): https://lightgbm.readthedocs.io/en/stable/Parameters.html
- CatBoost parameter tuning guide (learning rate, depth, l2_leaf_reg,
  overfitting detector / od_wait, class weights):
  https://catboost.ai/en/docs/concepts/parameter-tuning
- J. S. Aguiar, "LightGBM with Simple Features" (Home Credit Default Risk,
  1st-place method; LightGBM configuration and feature-engineering style):
  https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
- O. Grellier, "Python target encoding for categorical features" (out-of-fold
  smoothed target encoding; smoothing formula adapted here):
  https://www.kaggle.com/code/ogrellier/python-target-encoding-for-categorical-features
- W. Koehrsen, "Start Here: A Gentle Introduction" (competition EDA and
  baseline feature-engineering patterns for the application table):
  https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction

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
4. Blend weight grid-searched on pooled out-of-fold AUC (v3 generalises this to
   all models).
5. Prediction sanity checks: fails on (near-)constant test probabilities (the classic
   cause of a ~0.50 leaderboard AUC) and warns on test/OOF distribution drift.
6. OOF predictions are saved to output/oof_predictions.csv for offline analysis.
7. Fold-specific LightGBM categorical maps are now also pandas>=4 safe (unseen
   categories are mapped to NaN explicitly before Categorical construction).
8. Optional env knobs: GCL_TRAIN_PATH (train file override), GCL_CAT_ITERATIONS,
   GCL_LGB_N_ESTIMATORS (smoke-test budgets).

## Validation status

- End-to-end smoke test passed (v3, three models) on a synthetic train file
  (tools/make_synthetic_train.py, built from test.csv rows with an injected weak
  signal): 5 folds x 2 seeds x {CatBoost, LightGBM, XGBoost} completed, 12
  CV-encoded categorical features added (6 columns x frequency + target), OOF AUC
  catboost=0.9133 / lightgbm=0.9052 / xgboost=0.9063, grid-searched blend weights
  catboost=0.70, lightgbm=0.05, xgboost=0.25, blended OOF AUC=0.9144, and a valid
  61,500-row rank-blended submission was written. This validates code correctness
  and that the pipeline learns; it says nothing about real-data performance.
- The test/OOF drift check compares rank-scale means, because the final test
  submission is rank-normalised (a raw-mean comparison would false-alarm).
- input/train.csv is still absent, so real-data training/CV cannot be rerun here.
  The historical v1 blended AUC of 0.756483 remains the best real-data reference;
  v3's encodings + third model should improve on it, but that must be verified
  locally with the official train.csv.
- No external data was used at any point (competition rules); upgrades adapt cited
  public techniques (see References) to the official application table only.

## Next steps

1. Put the official train.csv in input/ locally; do not commit course data.
2. Install requirements.txt and run `python final_submission.py` (full budget:
   ~5k CatBoost iters, ~4k LightGBM trees with early stopping; on 2 cores expect
   roughly 1-3 hours. Set SEEDS=(42,) for a faster run).
3. Review fold AUCs and the pooled OOF AUC; compare with the v1 baseline on the same
   folds/seed before changing anything else.
4. Check output/submission.csv locally (row count, ID order, non-constant values)
   and upload the intended file - the last submission counts.

