# Home Credit Default Risk — GCI Global 2026

This project predicts the probability that a loan applicant experiences payment difficulties. The competition is evaluated with ROC AUC, so the model must output probabilities, not just 0/1 labels.

## Current pipeline

- Feature engineering for loan amounts, income, age/employment dates, external-source scores, missingness, and selected ratios.
- Five-fold stratified cross-validation with out-of-fold predictions.
- CatBoost + LightGBM probability blend.
- Fold-specific LightGBM categorical mappings and validation-based early stopping.
- Submission checks for row count, identifier order, required columns, and finite predictions.

## Project setup

Use Python 3.10+ (Python 3.11 is a sensible choice for compatibility).

1. Put the official competition files here:
   - `input/train.csv`
   - `input/test.csv`
   - `input/sample_submission.csv`
2. Install dependencies:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\\Scripts\\Activate.ps1
   # macOS/Linux:
   # source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run training from the repository root:

   ```bash
   python final_submission.py
   ```

The generated file is `output/submission.csv`.

## Evaluation and improvement

The script prints CatBoost, LightGBM, and blended out-of-fold ROC AUC. Treat this as a local validation estimate, not a guarantee of leaderboard performance. Compare experiments using the same folds and seeds. Do not choose model settings by repeatedly optimizing against the public leaderboard.

## Data and competition rules

- The dataset is not included in this repository. Obtain it from the official course/competition materials.
- Do not commit private or course-provided data files.
- Use only the competition-provided datasets; the assignment prohibits external data and hand-labeled predictions.
- Cite external code or notebooks you reference, as required by the competition rules.
- Keep a record of the exact script and settings used to generate each submitted CSV.

## Known limitations

- Training and validation have not been rerun in this environment because the official `input/train.csv` is absent from the repository.
- The current blend weights are a starting point, not proven optimal.
- A higher local AUC does not guarantee a higher public or private leaderboard score.
