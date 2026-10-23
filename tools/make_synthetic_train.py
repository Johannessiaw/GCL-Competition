"""Generate a synthetic train.csv that mimics the competition schema.

WHY: input/train.csv is course-provided and absent from this repository, so
the pipeline cannot be smoke-tested on real labels. This script builds a
train-shaped CSV from input/test.csv rows and injects a weak, KNOWN signal
(lower EXT_SOURCE_2 and younger age -> higher default risk). A healthy
pipeline must score clearly above 0.50 AUC on this file.

It is for pipeline validation only - NEVER commit or submit anything derived
from it, and never point GCL_TRAIN_PATH at it for a real submission.

Usage:
    python tools/make_synthetic_train.py
    GCL_TRAIN_PATH=input/train_synthetic.csv python final_submission.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_PATH = BASE_DIR / "input" / "test.csv"
OUT_PATH = BASE_DIR / "input" / "train_synthetic.csv"

N_ROWS = 12000
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    test = pd.read_csv(TEST_PATH)
    rows = test.sample(n=N_ROWS, replace=True, random_state=SEED).reset_index(drop=True)
    rows["SK_ID_CURR"] = np.arange(100_000, 100_000 + N_ROWS)

    ext2 = pd.to_numeric(rows["EXT_SOURCE_2"], errors="coerce")
    ext2 = ext2.fillna(ext2.median())
    age_years = -pd.to_numeric(rows["DAYS_BIRTH"], errors="coerce") / 365.25

    z_ext = (ext2 - ext2.mean()) / (ext2.std() + 1e-9)
    z_age = (age_years - age_years.mean()) / (age_years.std() + 1e-9)
    logit = -2.4 - 2.2 * z_ext + 0.9 * z_age + rng.normal(0.0, 0.6, N_ROWS)
    prob = 1.0 / (1.0 + np.exp(-logit))
    rows["TARGET"] = (rng.random(N_ROWS) < prob).astype(int)

    rows.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {N_ROWS:,} rows; default rate {rows['TARGET'].mean():.4%}")


if __name__ == "__main__":
    main()
