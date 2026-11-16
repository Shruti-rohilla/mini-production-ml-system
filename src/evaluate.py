import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

from data_pipeline import clean_data, load_data, split_data


def recall_at_precision(y_true, y_prob, min_precision=0.6):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(name, model, X, y, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    print(f"--- {name} (threshold={threshold:.3f}) ---")
    print(f"PR-AUC           : {average_precision_score(y, prob):.4f}")
    print(f"F1-macro         : {f1_score(y, pred, average='macro'):.4f}")
    print(f"Recall@Prec>=0.6 : {recall_at_precision(y, prob):.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y, pred))
    print()
    return prob


if __name__ == "__main__":
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    baseline = joblib.load("artifacts/baseline.joblib")
    lgbm = joblib.load("artifacts/lgbm.joblib")
    thr = joblib.load("artifacts/thresholds.joblib")

    print("=== TEST SET ===")
    base_prob = evaluate("Baseline (LogReg)", baseline, X_test, y_test, thr["baseline_threshold"])
    lgbm_prob = evaluate("LightGBM", lgbm, X_test, y_test, thr["lgbm_threshold"])

    # PR curve plot
    plt.figure(figsize=(6, 5))
    for label, prob in [("Baseline (LogReg)", base_prob), ("LightGBM", lgbm_prob)]:
        p, r, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.plot(r, p, label=f"{label} (AP={ap:.3f})")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", label="Random (churn rate)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/pr_curve.png", dpi=150)
    print("Saved artifacts/pr_curve.png")