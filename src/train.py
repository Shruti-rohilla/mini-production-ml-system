import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import NUMERIC_COLS, clean_data, load_data, split_data


def build_preprocessor(X_train):
    categorical_cols = [c for c in X_train.columns if c not in NUMERIC_COLS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def best_f1_threshold(y_true, y_prob):
    """Validation par wo threshold dhoondo jo F1 (churn class) maximize kare."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = np.argmax(f1[:-1])
    return float(thresholds[best])


def report(name, model, X, y, threshold=0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    print(
        f"{name:<22} PR-AUC={average_precision_score(y, prob):.4f}  "
        f"F1-macro={f1_score(y, pred, average='macro'):.4f}  "
        f"threshold={threshold:.3f}"
    )


if __name__ == "__main__":
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocessor: SIRF train par fit
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Baseline: Logistic Regression
    baseline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    baseline.fit(X_train, y_train)

    # Main model: LightGBM (imbalance ke liye scale_pos_weight)
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    lgbm = Pipeline([
        ("prep", preprocessor),
        ("clf", LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=7,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5.0,
            scale_pos_weight=spw, random_state=42, verbose=-1,
        )),
    ])
    lgbm.fit(X_train, y_train)

    # Threshold validation set par tune
    base_thr = best_f1_threshold(y_val, baseline.predict_proba(X_val)[:, 1])
    lgbm_thr = best_f1_threshold(y_val, lgbm.predict_proba(X_val)[:, 1])

    print("=== VALIDATION ===")
    report("Baseline (LogReg)", baseline, X_val, y_val, base_thr)
    report("LightGBM", lgbm, X_val, y_val, lgbm_thr)

    # Save
    joblib.dump(baseline, "artifacts/baseline.joblib")
    joblib.dump(lgbm, "artifacts/lgbm.joblib")
    joblib.dump({"lgbm_threshold": lgbm_thr, "baseline_threshold": base_thr},
                "artifacts/thresholds.joblib")
    print("\nSaved to artifacts/")