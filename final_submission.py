"""
Home Credit Default Risk - GCI Global 2026 (pipeline v3).

Upgrades over v2:
  1. Leakage-safe frequency and smoothed target encoding for categorical
     features (out-of-fold, after Ogrrellier's target-encoding kernel).
  2. Optional third model: XGBoost (hist, categorical-aware) - auto-skipped
     when the package is not installed (GCL_USE_XGB=0 disables).
  3. Blend weights grid-searched over all models on pooled OOF AUC; the
     test submission uses a rank-normalised weighted blend.
  4. All v2 upgrades retained: expanded features, tuned CatBoost/LightGBM
     with early stopping, multi-seed averaging, sanity checks, OOF artifact.

Data rules: only the official competition files are used. No external data
and no hand-labelled predictions (see README.md / MODEL_UPGRADE_NOTES.md).
Techniques are modelled on cited public sources; see MODEL_UPGRADE_NOTES.md.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # XGBoost is an optional third model
    XGBOOST_AVAILABLE = False

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
XGB_N_ESTIMATORS = int(os.environ.get("GCL_XGB_N_ESTIMATORS", 4000))
XGB_EARLY_STOP = int(os.environ.get("GCL_XGB_EARLY_STOP", 200))
USE_XGB = XGBOOST_AVAILABLE and os.environ.get("GCL_USE_XGB", "1") == "1"
TE_SMOOTHING = float(os.environ.get("GCL_TE_SMOOTHING", 10.0))


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



def rank_normalise(preds: np.ndarray) -> np.ndarray:
    """Map predictions to uniform [0, 1] ranks; standard Kaggle blend trick."""
    ranks = rankdata(preds)
    return (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)


def optimize_blend_weights(model_oof: dict[str, np.ndarray], y: pd.Series) -> tuple[dict[str, float], float]:
    """Grid-search non-negative blend weights (summing to 1) on pooled OOF AUC."""
    names = list(model_oof)
    if len(names) == 1:
        return {names[0]: 1.0}, roc_auc_score(y, model_oof[names[0]])

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
    best_weights, best_auc = {names[0]: 1.0}, -1.0
    if len(names) == 2:
        for w in grid:
            weights = {names[0]: float(w), names[1]: float(1.0 - w)}
            auc = roc_auc_score(y, sum(weights[m] * model_oof[m] for m in names))
            if auc > best_auc:
                best_auc, best_weights = auc, weights
    else:
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                weights = {names[0]: float(w1), names[1]: float(w2), names[2]: float(max(w3, 0.0))}
                auc = roc_auc_score(y, sum(weights[m] * model_oof[m] for m in names))
                if auc > best_auc:
                    best_auc, best_weights = auc, weights
    total = sum(best_weights.values())
    best_weights = {m: w / total for m, w in best_weights.items()}
    return best_weights, best_auc


def add_cv_encoded_features(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_features: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Leakage-safe frequency and smoothed target encoding for categoricals.

    Target encodings are computed out-of-fold (a row is encoded with target
    statistics from rows outside its own encoding fold); test rows use
    full-train statistics with the training prior as fallback. Technique after
    https://www.kaggle.com/code/ogrellier/python-target-encoding-for-categorical-features
    """
    X = X.copy()
    X_test = X_test.copy()
    added_cols = []
    prior = float(y.mean())
    enc_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cat_features:
        nunique = X[col].nunique()
        if nunique < 3 or nunique > 100:
            continue
        train_str = X[col].astype(str)
        test_str = X_test[col].astype(str)

        freq = train_str.value_counts(normalize=True)
        X[f"FE_{col}"] = train_str.map(freq).fillna(0.0).astype("float32")
        X_test[f"FE_{col}"] = test_str.map(freq).fillna(0.0).astype("float32")
        added_cols.append(f"FE_{col}")

        te_oof = pd.Series(np.nan, index=X.index, dtype="float64")
        for enc_train_idx, enc_valid_idx in enc_cv.split(X, y):
            fold_target = y.iloc[enc_train_idx]
            fold_str = train_str.iloc[enc_train_idx]
            sums = fold_target.groupby(fold_str).sum()
            counts = fold_target.groupby(fold_str).count()
            enc = (sums + prior * TE_SMOOTHING) / (counts + TE_SMOOTHING)
            te_oof.iloc[enc_valid_idx] = train_str.iloc[enc_valid_idx].map(enc)

        sums_full = y.groupby(train_str).sum()
        counts_full = y.groupby(train_str).count()
        enc_full = (sums_full + prior * TE_SMOOTHING) / (counts_full + TE_SMOOTHING)
        X[f"TE_{col}"] = te_oof.fillna(prior).astype("float32")
        X_test[f"TE_{col}"] = test_str.map(enc_full).fillna(prior).astype("float32")
        added_cols.append(f"TE_{col}")

    return X, X_test, added_cols


def fit_xgboost_fold(X_train_lgb, y_train, X_valid_lgb, y_valid, X_test_lgb, seed):
    """Train one XGBoost fold (hist, categorical-aware) with early stopping."""
    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        scale_pos_weight=neg / pos,
        tree_method="hist",
        random_state=seed,
        n_jobs=THREADS,
        verbosity=0,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=XGB_EARLY_STOP,
    )
    model.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        verbose=False,
    )
    valid_pred = model.predict_proba(X_valid_lgb)[:, 1]
    test_pred = model.predict_proba(X_test_lgb)[:, 1]
    best_iter = getattr(model, "best_iteration", None)
    best_iter = int(best_iter) + 1 if best_iter is not None else XGB_N_ESTIMATORS
    return valid_pred, test_pred, best_iter


def fit_blended_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_features: list[str]):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    model_names = ["catboost", "lightgbm"] + (["xgboost"] if USE_XGB else [])
    model_oof = {name: np.zeros(len(X), dtype=float) for name in model_names}
    model_test = {name: np.zeros(len(X_test), dtype=float) for name in model_names}

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

        fold_valid: dict[str, np.ndarray] = {}
        fold_test: dict[str, np.ndarray] = {}
        fold_iters: dict[str, int] = {}
        for seed in SEEDS:
            vp, tp, iters = fit_catboost_fold(
                X_train, y_train, X_valid, y_valid, X_test, cat_features, seed
            )
            fold_valid["catboost"] = vp if "catboost" not in fold_valid else fold_valid["catboost"] + vp
            fold_test["catboost"] = tp if "catboost" not in fold_test else fold_test["catboost"] + tp
            fold_iters["catboost"] = iters

            vp, tp, iters = fit_lightgbm_fold(
                X_train_lgb, y_train, X_valid_lgb, y_valid, X_test_lgb, cat_features, seed
            )
            fold_valid["lightgbm"] = vp if "lightgbm" not in fold_valid else fold_valid["lightgbm"] + vp
            fold_test["lightgbm"] = tp if "lightgbm" not in fold_test else fold_test["lightgbm"] + tp
            fold_iters["lightgbm"] = iters

            if USE_XGB:
                vp, tp, iters = fit_xgboost_fold(
                    X_train_lgb, y_train, X_valid_lgb, y_valid, X_test_lgb, seed
                )
                fold_valid["xgboost"] = vp if "xgboost" not in fold_valid else fold_valid["xgboost"] + vp
                fold_test["xgboost"] = tp if "xgboost" not in fold_test else fold_test["xgboost"] + tp
                fold_iters["xgboost"] = iters

        n_seeds = len(SEEDS)
        for name in model_names:
            fold_valid[name] /= n_seeds
            fold_test[name] /= n_seeds
            model_oof[name][idx_valid] = fold_valid[name]
            model_test[name] += fold_test[name] / cv.n_splits

        fold_auc = {name: roc_auc_score(y_valid, fold_valid[name]) for name in model_names}
        equal_blend = np.mean([fold_valid[name] for name in model_names], axis=0)
        fold_auc["blend"] = roc_auc_score(y_valid, equal_blend)
        fold_scores.append(fold_auc)
        parts = " | ".join(
            f"{name.capitalize()} AUC={fold_auc[name]:.6f} (iters~{fold_iters[name]})"
            for name in model_names
        )
        print(f"Fold {fold_number}/{cv.n_splits} | {parts} | Equal-weight Blend AUC={fold_auc['blend']:.6f}")

    pooled_auc = {name: roc_auc_score(y, model_oof[name]) for name in model_names}
    weights, blend_auc = optimize_blend_weights(model_oof, y)

    for name in model_names:
        print(f"Cross-validated {name.capitalize()} AUC: {pooled_auc[name]:.6f}")
    print("OOF-optimized blend weights: " + ", ".join(f"{m}={w:.2f}" for m, w in weights.items()))
    print(f"Cross-validated blended AUC: {blend_auc:.6f}")
    print(
        "Fold blend AUC range: "
        f"{min(row['blend'] for row in fold_scores):.6f} - "
        f"{max(row['blend'] for row in fold_scores):.6f}"
    )

    blended_oof = sum(weights[m] * model_oof[m] for m in model_names)
    # Rank-normalised weighted blend for the test set (calibration-robust).
    test_probs = sum(weights[m] * rank_normalise(model_test[m]) for m in model_names)
    return model_oof, blended_oof, test_probs, weights


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
    # Compare on the rank scale: test probabilities are rank-normalised in the
    # final blend, so raw-probability means are not comparable to OOF means.
    drift = abs(float(rank_normalise(probs).mean()) - float(rank_normalise(oof_probs).mean()))
    if drift > 0.05:
        print(
            f"WARNING: test/OOF rank-mean drift of {drift:.3f}; "
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

    X, X_test, enc_cols = add_cv_encoded_features(X, y, X_test, cat_features)
    if enc_cols:
        print(f"Added {len(enc_cols)} CV-encoded categorical features (frequency + target)")

    model_oofs, blended_oof, probs, weights = fit_blended_models(X, y, X_test, cat_features)
    oof_df = pd.DataFrame({
        "SK_ID_CURR": train_df["SK_ID_CURR"].to_numpy(),
        "TARGET": y.to_numpy(),
    })
    for name, oof in model_oofs.items():
        oof_df[f"{name}_oof"] = oof
    oof_df["blended_oof"] = blended_oof

    if len(probs) != len(test_df):
        raise ValueError("Prediction count does not match test data row count.")
    if not np.isfinite(probs).all():
        raise ValueError("The model produced non-finite prediction probabilities.")

    prediction_sanity_checks(probs, blended_oof)

    OUTPUT_DIR.mkdir(exist_ok=True)
    oof_df.to_csv(OOF_PATH, index=False)
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




