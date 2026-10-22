# Model upgrade notes

## Changes applied to main

1. LightGBM categorical columns use category maps built from each fold's training rows and reused unchanged for validation and test.
2. LightGBM uses a validation fold and early stopping rather than always fitting the full preset number of trees.
3. The training script reports CatBoost, LightGBM, and blended ROC-AUC for each fold and pooled out-of-fold predictions.
4. Startup checks validate required files, non-empty data, duplicate column names, missing/duplicate IDs, binary target labels, exact submission columns/order, train/test feature-schema agreement, sample-submission row count and ID order, and finite output probabilities.
5. Repository instructions explain reproducibility, competition restrictions, and the local-data requirement.
6. The public main branch no longer contains the generated output/submission.csv.

## Competition criteria considered

- Metric: ROC-AUC, requiring probability predictions rather than hard class labels.
- Submission: exactly SK_ID_CURR,TARGET, matching the sample submission's rows and identifier order.
- Reproducibility: fixed random seeds and code capable of recreating the submitted CSV.
- Data rules: use only the official provided data; no external data or hand-labelled test predictions.
- Final result: the assignment says the last submitted file determines the final ranking, so verify the intended file before submitting.

## Important verification status

The repository has input/test.csv and input/sample_submission.csv, but input/train.csv is absent. The model cannot be honestly benchmarked or retrained from the repository alone until the official training file is made available locally. No improvement in competition score is claimed yet. The historical blended AUC in run_model.log is 0.756483; it is a prior run, not a verified result for the updated script.

## Next steps

1. Put the official train.csv in input/ locally; do not commit course data.
2. Install requirements.txt in a compatible Python environment and run python final_submission.py.
3. Review all five fold scores and pooled OOF AUC, compare fairly with the old baseline using the same folds/seed, and only then consider alternative blend weights.
4. Check the generated CSV locally and upload the intended file to Omnicampus. Since the last submission counts, avoid submitting an unverified replacement.
