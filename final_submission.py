from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
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


def make_lgbm_dataset(
    df: pd.DataFrame,
    cat_features: list[str],
    category_maps: dict[str, pd.Index],
) -> pd.DataFrame:
    """Apply identical categorical ordering to training, validation and test rows."""
    out = df.copy()
    for col in cat_features:
        values = out[col].fillna("Missing").astype(str)
        # Unseen validation/test categories become NaN without shifting category codes.
        out[col] = pd.Categorical(values, categories=category_maps[col])
    return out


def fit_blended_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, cat_features: list[str]):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_oof = np.zeros(len(X), dtype=float)
    lgb_oof = np.zeros(len(X), dtype=float)
    cat_test = np.zeros(len(X_test), dtype=float)
    lgb_test = np.zeros(len(X_test), dtype=float)

    fold_scores = []
    for fold_number, (idx_train, idx_valid) in enumerate(cv.split(X, y), start=1):
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
        cat_valid_pred = cat_model.predict_proba(X_valid)[:, 1]
        cat_oof[idx_valid] = cat_valid_pred
        cat_test += cat_model.predict_proba(X_test)[:, 1] / cv.n_splits
        cat_fold_auc = roc_auc_score(y_valid, cat_valid_pred)

        # Build category maps from this fold's training rows only, then reuse them.
        category_maps = {
            col: pd.Index(sorted(X_train[col].fillna("Missing").astype(str).unique()))
            for col in cat_features
        }
        X_train_lgb = make_lgbm_dataset(X_train, cat_features, category_maps)
        X_valid_lgb = make_lgbm_dataset(X_valid, cat_features, category_maps)
        X_test_lgb = make_lgbm_dataset(X_test, cat_features, category_maps)

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        lgb_model = LGBMClassifier(
            n_estimators=3000,
            objective="binary",
            num_leaves=31,
            learning_rate=0.025,
            min_child_samples=80,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=5.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        fit_kwargs = {
            "eval_set": [(X_valid_lgb, y_valid)],
            "eval_metric": "auc",
            "callbacks": [early_stopping(120, verbose=False), log_evaluation(0)],
        }
        if cat_features:
            lgb_model.fit(
                X_train_lgb, y_train, categorical_feature=cat_features, **fit_kwargs
            )
        else:
            lgb_model.fit(X_train_lgb, y_train, **fit_kwargs)
        lgb_valid_pred = lgb_model.predict_proba(X_valid_lgb)[:, 1]
        lgb_oof[idx_valid] = lgb_valid_pred
        lgb_test += lgb_model.predict_proba(X_test_lgb)[:, 1] / cv.n_splits
        lgb_fold_auc = roc_auc_score(y_valid, lgb_valid_pred)
        blend_fold_auc = roc_auc_score(y_valid, 0.55 * cat_valid_pred + 0.45 * lgb_valid_pred)
        fold_scores.append({
            "fold": fold_number,
            "catboost_auc": cat_fold_auc,
            "lightgbm_auc": lgb_fold_auc,
            "blend_auc": blend_fold_auc,
        })
        print(
            f"Fold {fold_number}/{cv.n_splits} | "
            f"CatBoost AUC={cat_fold_auc:.6f} | "
            f"LightGBM AUC={lgb_fold_auc:.6f} | "
            f"Blend AUC={blend_fold_auc:.6f}"
        )

    cat_auc = roc_auc_score(y, cat_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    blended_oof = 0.55 * cat_oof + 0.45 * lgb_oof
    blended_auc = roc_auc_score(y, blended_oof)
    print(f"Cross-validated CatBoost AUC: {cat_auc:.6f}")
    print(f"Cross-validated LightGBM AUC: {lgb_auc:.6f}")
    print(f"Cross-validated blended AUC: {blended_auc:.6f}")
    print(
        "Fold blend AUC range: "
        f"{min(row['blend_auc'] for row in fold_scores):.6f} - "
        f"{max(row['blend_auc'] for row in fold_scores):.6f}"
    )
    return 0.55 * cat_test + 0.45 * lgb_test


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

    print(f"Training rows: {len(train_df):,}; test rows: {len(test_df):,}")
    print(f"Input feature columns: {len(train_features):,}")
    print(f"Default rate: {train_df['TARGET'].mean():.6%}")
    print("Validation: 5-fold stratified CV, shuffle=True, random_state=42")

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    X, y, X_test, cat_features = prepare_model_data(train_df, test_df)
    probs = fit_blended_models(X, y, X_test, cat_features)

    if len(probs) != len(test_df):
        raise ValueError("Prediction count does not match test data row count.")
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
