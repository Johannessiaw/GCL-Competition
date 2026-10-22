# Model upgrade notes

## Changes in this branch

1. LightGBM categorical columns now use category maps built from each fold's training rows and reused unchanged for validation and test. Previously, categories were inferred independently per split, which could create inconsistent category codes.
2. LightGBM now uses a validation fold and early stopping instead of always fitting the full preset number of trees.
3. The training script prints separate out-of-fold AUC scores for CatBoost, LightGBM, and the blend.
4. Startup checks give clear errors when the required input files or columns are missing.
5. Repository setup instructions and dependency pins/ranges are documented; course data and generated submissions are ignored by Git.

## What has not been proven

The changes have not been benchmarked against the real training data in this environment because `input/train.csv` is not present in the repository. No improvement in leaderboard score is claimed until the updated script is run and its validation AUC and submission are checked.

## Next experiment plan

1. Run the upgraded script and save all three AUC values.
2. Compare with the previous baseline using the same five-fold split and random seed.
3. Inspect ROC AUC per fold to check stability, not only the pooled out-of-fold score.
4. Try blend weights only after both individual models are evaluated.
5. Submit a valid CSV and record the timestamp and score. The course instructions say the last submission determines the final result, so verify the intended file before submitting.
