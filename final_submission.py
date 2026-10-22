from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "input" / "train.csv"
TEST_PATH = BASE_DIR / "input" / "test.csv"
SAMPLE_SUB_PATH = BASE_DIR / "input" / "sample_submission.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"


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
        "CNT_FAM_MEMBERS",
        "REGION_POPULATION_RELATIVE",
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

    for col, feature_name in [
        ("DAYS_BIRTH", "AGE_YEARS"),
        ("DAYS_EMPLOYED", "YEARS_EMPLOYED"),
        ("DAYS_REGISTRATION", "YEARS_REGISTRATION"),
        ("DAYS_ID_PUBLISH", "YEARS_ID_PUBLISH"),
        ("DAYS_LAST_PHONE_CHANGE", "YEARS_PHONE_CHANGE"),
    ]:
        if col in df.columns:
            df[feature_name] = -df[col] / 365.25

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
            X[col] = X[col].fillna("Missing")
            X_test[col] = X_test[col].fillna("Missing")

    cat_features = list(X.select_dtypes(include=["object", "string"]).columns)
    if not cat_features:
        cat_features = []

    return X, y, X_test, cat_features


def make_lgbm_dataset(df: pd.DataFrame, cat_features: list[str], reference_df: pd.DataFrame | None = None):
    out = df.copy()
    if not cat_features:
        return out
    all_values = pd.concat(
        [out[col].astype(str) for col in cat_features] + ([reference_df[col].astype(str) for col in cat_features] if reference_df is not None else []),
        axis=1,
    )
    for col in cat_features:
        categories = pd.Index(pd.concat([out[col].astype(str), reference_df[col].astype(str)], ignore_index=True).unique()) if reference_df is not None else pd.Index(out[col].astype(str).unique())
        out[col] = pd.Categorical(out[col].astype(str), categories=categories)
    return out


def fit_blended_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_features: list[str]):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_oof = np.zeros(len(X), dtype=float)
    lgb_oof = np.zeros(len(X), dtype=float)
    cat_test = np.zeros(len(X_test), dtype=float)
    lgb_test = np.zeros(len(X_test), dtype=float)

    for idx_train, idx_valid in cv.split(X, y):
        X_train, X_valid = X.iloc[idx_train].copy(), X.iloc[idx_valid].copy()
        y_train, y_valid = y.iloc[idx_train].copy(), y.iloc[idx_valid].copy()

        cat_model = CatBoostClassifier(
            iterations=1400,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=2.5,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features if cat_features else None,
            auto_class_weights="Balanced",
            subsample=0.9,
            colsample_bylevel=0.9,
            thread_count=8,
            od_type="Iter",
            od_wait=60,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=120, verbose=False)
        cat_oof[idx_valid] = cat_model.predict_proba(X_valid)[:, 1]
        cat_test += cat_model.predict_proba(X_test)[:, 1] / cv.n_splits

        X_train_lgb = make_lgbm_dataset(X_train, cat_features, X_test)
        X_valid_lgb = make_lgbm_dataset(X_valid, cat_features, X_test)
        X_test_lgb = make_lgbm_dataset(X_test, cat_features, X_test)

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        lgb_model = LGBMClassifier(
            n_estimators=2200,
            objective="binary",
            num_leaves=31,
            learning_rate=0.03,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=5.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=8,
            verbosity=-1,
        )
        if cat_features:
            lgb_model.fit(X_train_lgb, y_train, categorical_feature=cat_features)
        else:
            lgb_model.fit(X_train_lgb, y_train)
        lgb_oof[idx_valid] = lgb_model.predict_proba(X_valid_lgb)[:, 1]
        lgb_test += lgb_model.predict_proba(X_test_lgb)[:, 1] / cv.n_splits

    blended_oof = 0.55 * cat_oof + 0.45 * lgb_oof
    blended_auc = roc_auc_score(y, blended_oof)
    print(f"Cross-validated blended AUC: {blended_auc:.6f}")
    return 0.55 * cat_test + 0.45 * lgb_test


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    X, y, X_test, cat_features = prepare_model_data(train_df, test_df)
    probs = fit_blended_models(X, y, X_test, cat_features)

    if len(sample_submission) != len(test_df):
        raise ValueError("Sample submission and test data have different row counts.")
    if not sample_submission["SK_ID_CURR"].equals(test_df["SK_ID_CURR"]):
        raise ValueError("Sample submission IDs do not match test data IDs in order.")
    if not np.isfinite(probs).all():
        raise ValueError("The model produced non-finite prediction probabilities.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        "SK_ID_CURR": sample_submission["SK_ID_CURR"],
        "TARGET": probs.astype(float),
    })
    submission.to_csv(OUTPUT_PATH, index=False)

    print("Saved submission to:", OUTPUT_PATH)
    print(f"Rows: {len(submission)}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
