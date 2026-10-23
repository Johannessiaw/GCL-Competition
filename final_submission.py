"""
Home Credit Default Risk - GCI Global 2026 (pipeline v2).

Upgrades over v1:
  1. Expanded feature engineering (loan-duration proxy, income-per-child,
     bureau-enquiry aggregates, EXT_SOURCE statistics/interactions,
     region-rating interactions, car-ownership indicators).
  2. Tuned CatBoost and LightGBM with validation-based early stopping.
  3. Multi-seed averaging per model for prediction stability.
  4. Blend weight chosen by grid search on out-of-fold AUC instead of a
     fixed 0.55/0.45 split.
  5. Prediction sanity checks (variance, distribution drift vs OOF) and an
     OOF artifact saved to output/ for offline analysis.

Data rules: only the official competition files are used. No external data
and no hand-labelled predictions (see README.md / MODEL_UPGRADE_NOTES.md).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = Path(os.environ.get("GCL_TRAIN_PATH", BASE_DIR / "input" / "train.csv"))
TEST_PATH = BASE_DIR / "input" / "test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "input" / "sample_submission.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"

N_SPLITS = 5
SEEDS = (42, 2026)  # set to (42,) for a faster run
THREADS = max(1, min(8, os.cpu_count() or 2))
CAT_OD_WAIT = 300
LGB_EARLY_STOP = 200
CAT_ITERATIONS = int(os.environ.get("GCL_CAT_ITERATIONS", 5000))
LGB_N_ESTIMATORS = int(os.environ.get("GCL_LGB_N_ESTIMATORS", 4000))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_columns = [col for col in df.columns if col not in ["TARGET", "SK_ID_CURR"]]

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce")
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] > 0).astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] <= 0, np.nan)

    numeric_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    ratios = [
        ("AMT_CREDIT_AMT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("AMT_ANNUITY_AMT_CREDIT_RATIO", "AMT_ANNUITY", "AMT_CREDIT"),
        ("AMT_GOODS_PRICE_AMT_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT"),
        ("AMT_ANNUITY_AMT_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("AMT_GOODS_PRICE_AMT_INCOME_RATIO", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"),
        ("DAYS_EMPLOYED_DAYS_BIRTH_RATIO", "DAYS_EMPLOYED", "DAYS_BIRTH"),
        ("AMT_INCOME_TOTAL_PER_FAMILY_MEMBER", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        ("DAYS_REGISTRATION_DAYS_BIRTH_RATIO", "DAYS_REGISTRATION", "DAYS_BIRTH"),
        ("DAYS_ID_PUBLISH_DAYS_BIRTH_RATIO", "DAYS_ID_PUBLISH", "DAYS_BIRTH"),
    ]
    for new_col, num_col, den_col in ratios:
        if num_col in df.columns and den_col in df.columns:
            denom = df[den_col].replace(0, np.nan)
            df[new_col] = df[num_col] / denom

    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ext_source_cols = [
        col for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if col in df.columns
    ]
    if ext_source_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_source_cols].mean(axis=1, skipna=True)
        df["EXT_SOURCE_MAX"] = df[ext_source_cols].max(axis=1, skipna=True)
        df["EXT_SOURCE_MIN"] = df[ext_source_cols].min(axis=1, skipna=True)
        df["EXT_SOURCE_SUM"] = df[ext_source_cols].sum(axis=1, skipna=True)
        df["EXT_SOURCE_STD"] = df[ext_source_cols].std(axis=1, skipna=True)
        df["EXT_SOURCE_COUNT"] = df[ext_source_cols].notna().sum(axis=1)

    for left, right in [
        ("EXT_SOURCE_1", "EXT_SOURCE_2"),
        ("EXT_SOURCE_1", "EXT_SOURCE_3"),
        ("EXT_SOURCE_2", "EXT_SOURCE_3"),
    ]:
        if left in df.columns and right in df.columns:
            df[f"{left}_{right}_PRODUCT"] = df[left] * df[right]
            df[f"{left}_{right}_DIFF"] = df[left] - df[right]
            df[f"{left}_{right}_RATIO"] = df[left] / df[right].replace(0, np.nan)

    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if col in df.columns:
            df[f"{col}_MISSING"] = df[col].isna().astype("int8")

    df["APPLICATION_MISSING_COUNT"] = df[base_columns].isna().sum(axis=1)

    for col in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "REGION_POPULATION_RELATIVE",
    ]:
        if col in df.columns:
            df[f"{col}_LOG1P"] = np.log1p(df[col].clip(lower=0))

    # ---- v2 additions -------------------------------------------------
    # Loan-duration proxy: how many annuity payments the credit implies.
    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_DIV_ANNUITY_MONTHS"] = (
            df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
        )

    # Household structure.
    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
        df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)
    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["CHILDREN_FAMILY_RATIO"] = (
            df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
        )

    # Region rating interactions.
    if {"REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE"}.issubset(df.columns):
        df["REGION_RATING_X_POPULATION"] = (
            df["REGION_RATING_CLIENT"] * df["REGION_POPULATION_RELATIVE"]
        )

    # Car ownership indicators.
    if "OWN_CAR_AGE" in df.columns:
        df["OWN_CAR_AGE_MISSING"] = df["OWN_CAR_AGE"].isna().astype("int8")
        if "DAYS_BIRTH" in df.columns:
            driver_age = (-df["DAYS_BIRTH"] / 365.25).replace(0, np.nan)
            df["CAR_AGE_VS_DRIVER_AGE"] = df["OWN_CAR_AGE"] / driver_age

    # Credit-bureau enquiry aggregates (recent enquiries weighted higher).
    bureau_cols = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    if bureau_cols:
        df["BUREAU_ENQUIRIES_TOTAL"] = df[bureau_cols].sum(axis=1, skipna=True)
        recency_weights = {"HOUR": 16, "MON": 8, "QRT": 4, "YEAR": 1}
        weighted = None
        for suffix, weight in recency_weights.items():
            col = f"AMT_REQ_CREDIT_BUREAU_{suffix}"
            if col in df.columns:
                term = weight * pd.to_numeric(df[col], errors="coerce").fillna(0)
                weighted = term if weighted is None else weighted + term
        if weighted is not None:
            df["BUREAU_ENQUIRIES_RECENCY_WEIGHTED"] = weighted

    # Document and contact-channel counts when present.
    flag_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if flag_cols:
        df["DOCUMENT_FLAGS_SUM"] = df[flag_cols].sum(axis=1)
    contact_cols = [
        c
        for c in [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        ]
        if c in df.columns
    ]
    if contact_cols:
        df["CONTACT_CHANNELS_SUM"] = df[contact_cols].sum(axis=1)

    # Recency features.
    if {"DAYS_REGISTRATION", "DAYS_ID_PUBLISH"}.issubset(df.columns):
        df["REGISTRATION_MINUS_ID_PUBLISH"] = df["DAYS_REGISTRATION"] - df["DAYS_ID_PUBLISH"]
    if {"DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"}.issubset(df.columns):
        df["PHONE_CHANGE_PER_YEAR_OF_AGE"] = (
            (-df["DAYS_LAST_PHONE_CHANGE"] / 365.25)
            / (-df["DAYS_BIRTH"] / 365.25).replace(0, np.nan)
        )

    # Social-circle default ratios when present.
    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        df["DEF_30_PER_OBS_30"] = (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"].replace(0, np.nan)
        )
    if {"DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"}.issubset(df.columns):
        df["DEF_60_PER_OBS_60"] = (
            df["DEF_60_CNT_SOCIAL_CIRCLE"] / df["OBS_60_CNT_SOCIAL_CIRCLE"].replace(0, np.nan)
        )

    # Age buckets.
    if "DAYS_BIRTH" in df.columns:
        age_years = -df["DAYS_BIRTH"] / 365.25
        df["AGE_YEARS"] = age_years
        df["AGE_UNDER_30"] = (age_years < 30).astype("int8")
        df["AGE_OVER_55"] = (age_years > 55).astype("int8")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].fillna("Missing")

    return df


def prepare_model_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_columns = [col for col in train_df.columns if col not in ["TARGET", "SK_ID_CURR"]]
    X = train_df[feature_columns].copy()
    y = train_df["TARGET"].astype("int32").copy()
    X_test = test_df[feature_columns].copy()

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Missing").astype(str)
            X_test[col] = X_test[col].fillna("Missing").astype(str)

    cat_features = list(X.select_dtypes(include=["object", "string"]).columns)

    return X, y, X_test, cat_features


def make_lgbm_dataset(
    df: pd.DataFrame,
    cat_features: list[str],
    category_maps: dict[str, pd.Index],
) -> pd.DataFrame:
    """Apply identical categorical ordering to training, validation and test rows."""
    out = df.copy()
    for col in cat_features:
        values = out[col].fillna("Missing").astype(str)
        # Unseen validation/test categories become NaN without shifting codes.
        # Map first so pandas>=4 does not raise on out-of-category values.
        mapped = values.where(values.isin(category_maps[col]))
        out[col] = pd.Categorical(mapped, categories=category_maps[col])
    return out


def fit_catboost_fold(X_train, y_train, X_valid, y_valid, X_test, cat_features, seed):
    """Train one CatBoost fold with early stopping; return valid/test probabilities."""
    model = CatBoostClassifier(
        iterations=CAT_ITERATIONS,
        depth=7,
        learning_rate=0.035,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat_features or None,
        auto_class_weights="Balanced",
        subsample=0.9,
        colsample_bylevel=0.9,
        thread_count=THREADS,
        od_type="Iter",
        od_wait=CAT_OD_WAIT,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    valid_pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    best_iter = (model.get_best_iteration() or 0) + 1
    return valid_pred, test_pred, best_iter


def fit_lightgbm_fold(X_train_lgb, y_train, X_valid_lgb, y_valid, X_test_lgb, cat_features, seed):
    """Train one LightGBM fold with early stopping; return valid/test probabilities."""
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = LGBMClassifier(
        n_estimators=LGB_N_ESTIMATORS,
        objective="binary",
        num_leaves=48,
        learning_rate=0.03,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=10.0,
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=THREADS,
        verbosity=-1,
    )
    fit_kwargs = {
        "eval_set": [(X_valid_lgb, y_valid)],
        "eval_metric": "auc",
        "callbacks": [early_stopping(LGB_EARLY_STOP, verbose=False), log_evaluation(0)],
    }
    if cat_features:
        model.fit(X_train_lgb, y_train, categorical_feature=cat_features, **fit_kwargs)
    else:
        model.fit(X_train_lgb, y_train, **fit_kwargs)
    valid_pred = model.predict_proba(X_valid_lgb)[:, 1]
    test_pred = model.predict_proba(X_test_lgb)[:, 1]
    best_iter = getattr(model, "best_iteration_", None) or LGB_N_ESTIMATORS
    return valid_pred, test_pred, best_iter



def optimize_blend_weight(cat_oof: np.ndarray, lgb_oof: np.ndarray, y: pd.Series) -> tuple[float, float]:
    """Grid-search the CatBoost weight that maximizes pooled OOF AUC."""
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 101):
        auc = roc_auc_score(y, w * cat_oof + (1.0 - w) * lgb_oof)
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    return best_w, best_auc


def fit_blended_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_features: list[str]):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    cat_oof = np.zeros(len(X), dtype=float)
    lgb_oof = np.zeros(len(X), dtype=float)
    cat_test = np.zeros(len(X_test), dtype=float)
    lgb_test = np.zeros(len(X_test), dtype=float)

    fold_scores = []
    for fold_number, (idx_train, idx_valid) in enumerate(cv.split(X, y), start=1):
        X_train, X_valid = X.iloc[idx_train].copy(), X.iloc[idx_valid].copy()
        y_train, y_valid = y.iloc[idx_train].copy(), y.iloc[idx_valid].copy()

        # Category maps built from this fold's training rows only, then reused.
        category_maps = {
            col: pd.Index(sorted(X_train[col].fillna("Missing").astype(str).unique()))
            for col in cat_features
        }
        X_train_lgb = make_lgbm_dataset(X_train, cat_features, category_maps)
        X_valid_lgb = make_lgbm_dataset(X_valid, cat_features, category_maps)
        X_test_lgb = make_lgbm_dataset(X_test, cat_features, category_maps)

        cat_valid_seeds, cat_test_seeds = [], []
        lgb_valid_seeds, lgb_test_seeds = [], []
        for seed in SEEDS:
            cat_vp, cat_tp, cat_iters = fit_catboost_fold(
                X_train, y_train, X_valid, y_valid, X_test, cat_features, seed
            )
            cat_valid_seeds.append(cat_vp)
            cat_test_seeds.append(cat_tp)

            lgb_vp, lgb_tp, lgb_iters = fit_lightgbm_fold(
                X_train_lgb, y_train, X_valid_lgb, y_valid, X_test_lgb, cat_features, seed
            )
            lgb_valid_seeds.append(lgb_vp)
            lgb_test_seeds.append(lgb_tp)

        cat_valid = np.mean(cat_valid_seeds, axis=0)
        cat_test_pred = np.mean(cat_test_seeds, axis=0)
        lgb_valid = np.mean(lgb_valid_seeds, axis=0)
        lgb_test_pred = np.mean(lgb_test_seeds, axis=0)

        cat_oof[idx_valid] = cat_valid
        lgb_oof[idx_valid] = lgb_valid
        cat_test += cat_test_pred / cv.n_splits
        lgb_test += lgb_test_pred / cv.n_splits

        cat_fold_auc = roc_auc_score(y_valid, cat_valid)
        lgb_fold_auc = roc_auc_score(y_valid, lgb_valid)
        blend_fold_auc = roc_auc_score(y_valid, 0.5 * cat_valid + 0.5 * lgb_valid)
        fold_scores.append({
            "fold": fold_number,
            "catboost_auc": cat_fold_auc,
            "lightgbm_auc": lgb_fold_auc,
            "blend_auc": blend_fold_auc,
        })
        print(
            f"Fold {fold_number}/{cv.n_splits} | "
            f"CatBoost AUC={cat_fold_auc:.6f} (iters~{cat_iters}) | "
            f"LightGBM AUC={lgb_fold_auc:.6f} (iters~{lgb_iters}) | "
            f"50/50 Blend AUC={blend_fold_auc:.6f}"
        )

    cat_auc = roc_auc_score(y, cat_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    blend_w, blended_auc = optimize_blend_weight(cat_oof, lgb_oof, y)

    print(f"Cross-validated CatBoost AUC:  {cat_auc:.6f}")
    print(f"Cross-validated LightGBM AUC:  {lgb_auc:.6f}")
    print(f"OOF-optimized blend weight (CatBoost): {blend_w:.2f}")
    print(f"Cross-validated blended AUC:   {blended_auc:.6f}")
    print(
        "Fold blend AUC range: "
        f"{min(row['blend_auc'] for row in fold_scores):.6f} - "
        f"{max(row['blend_auc'] for row in fold_scores):.6f}"
    )

    return cat_oof, lgb_oof, blend_w * cat_test + (1.0 - blend_w) * lgb_test


def prediction_sanity_checks(probs: np.ndarray, oof_probs: np.ndarray) -> None:
    """Fail loudly on degenerate predictions; warn on test/OOF distribution drift."""
    std = float(np.std(probs))
    print(
        f"Test prediction stats: mean={probs.mean():.5f} std={std:.5f} "
        f"min={probs.min():.5f} max={probs.max():.5f}"
    )
    print(
        f"OOF prediction stats:  mean={float(oof_probs.mean()):.5f} "
        f"std={float(oof_probs.std()):.5f}"
    )
    if std < 1e-6:
        raise ValueError(
            "Test predictions are (near-)constant; such a submission scores ~0.50 AUC."
        )
    drift = abs(float(probs.mean()) - float(oof_probs.mean()))
    if drift > 0.15:
        print(
            f"WARNING: test/OOF mean drift of {drift:.3f}; "
            "inspect for schema or distribution mismatch before submitting."
        )



def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_PATH}. "
            "Download the official train.csv and place it in the input/ folder; "
            "do not commit the dataset to Git."
        )
    for required_path in (TEST_PATH, SAMPLE_SUB_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Required competition file not found: {required_path}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

    if train_df.empty or test_df.empty or sample_submission.empty:
        raise ValueError("One or more competition CSV files are empty.")
    if train_df.columns.duplicated().any() or test_df.columns.duplicated().any():
        raise ValueError("Training or test data contains duplicate column names.")

    required_train_columns = {"SK_ID_CURR", "TARGET"}
    if not required_train_columns.issubset(train_df.columns):
        raise ValueError(f"train.csv must contain columns: {sorted(required_train_columns)}")
    if "SK_ID_CURR" not in test_df.columns:
        raise ValueError("test.csv must contain the SK_ID_CURR identifier column")
    if not {"SK_ID_CURR", "TARGET"}.issubset(sample_submission.columns):
        raise ValueError("sample_submission.csv must contain SK_ID_CURR and TARGET columns")
    if train_df["SK_ID_CURR"].isna().any() or test_df["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR contains missing values.")
    if train_df["SK_ID_CURR"].duplicated().any() or test_df["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR must be unique within train.csv and test.csv.")
    if train_df["TARGET"].isna().any() or not set(train_df["TARGET"].unique()).issubset({0, 1}):
        raise ValueError("TARGET must contain only non-missing binary labels 0 and 1.")
    if not sample_submission.columns.tolist() == ["SK_ID_CURR", "TARGET"]:
        raise ValueError(
            "Submission template must contain exactly these columns in order: "
            "SK_ID_CURR, TARGET."
        )
    if len(sample_submission) != len(test_df):
        raise ValueError("Sample submission and test data have different row counts.")
    if not sample_submission["SK_ID_CURR"].equals(test_df["SK_ID_CURR"]):
        raise ValueError("Sample submission IDs do not match test data IDs in order.")

    train_features = set(train_df.columns) - {"TARGET", "SK_ID_CURR"}
    test_features = set(test_df.columns) - {"SK_ID_CURR"}
    missing_test = sorted(train_features - test_features)
    extra_test = sorted(test_features - train_features)
    if missing_test or extra_test:
        raise ValueError(
            "Train/test feature mismatch. "
            f"Missing from test: {missing_test[:10]}; "
            f"extra in test: {extra_test[:10]}"
        )

    print(f"Training data: {TRAIN_PATH}")
    print(f"Training rows: {len(train_df):,}; test rows: {len(test_df):,}")
    print(f"Input feature columns: {len(train_features):,}")
    print(f"Default rate: {train_df['TARGET'].mean():.6%}")
    print(
        f"Validation: {N_SPLITS}-fold stratified CV, shuffle=True, random_state=42; "
        f"seeds per model: {SEEDS}"
    )

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    X, y, X_test, cat_features = prepare_model_data(train_df, test_df)
    print(f"Modelled features: {X.shape[1]:,} (categorical: {len(cat_features)})")

    cat_oof, lgb_oof, probs = fit_blended_models(X, y, X_test, cat_features)
    blended_oof = pd.concat(
        [train_df["SK_ID_CURR"], y, pd.Series(cat_oof, name="catboost_oof"),
         pd.Series(lgb_oof, name="lightgbm_oof")],
        axis=1,
    )

    if len(probs) != len(test_df):
        raise ValueError("Prediction count does not match test data row count.")
    if not np.isfinite(probs).all():
        raise ValueError("The model produced non-finite prediction probabilities.")

    prediction_sanity_checks(probs, blended_oof[["catboost_oof", "lightgbm_oof"]].mean(axis=1).to_numpy())

    OUTPUT_DIR.mkdir(exist_ok=True)
    blended_oof.to_csv(OOF_PATH, index=False)
    submission = pd.DataFrame({
        "SK_ID_CURR": sample_submission["SK_ID_CURR"],
        "TARGET": probs.astype(float),
    })
    submission.to_csv(OUTPUT_PATH, index=False)

    print("Saved OOF predictions to:", OOF_PATH)
    print("Saved submission to:", OUTPUT_PATH)
    print(f"Rows: {len(submission)}")
    print(submission.head().to_string(index=False))
    print("Reminder: verify this is the intended file before uploading - the last submission counts.")


if __name__ == "__main__":
    main()




