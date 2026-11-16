import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
PRIOR_CHURN_RATE = 0.2654

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("predict_batch")


def read_input(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main():
    parser = argparse.ArgumentParser(description="Batch churn scoring")
    parser.add_argument("--input", required=True, help="CSV or XLSX file")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--id-col", default="CustomerID")
    args = parser.parse_args()

    model = joblib.load(ARTIFACTS / "lgbm.joblib")
    threshold = joblib.load(ARTIFACTS / "thresholds.joblib")["lgbm_threshold"]
    required = list(model.named_steps["prep"].feature_names_in_)

    df = read_input(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing required columns: %s", missing)
        sys.exit(1)

    if args.id_col in df.columns:
        ids = df[args.id_col]
    else:
        log.warning("ID column '%s' not found, using row index", args.id_col)
        ids = pd.Series(df.index, name=args.id_col)

    X = df[required].copy()
    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce").fillna(0.0)

    start = time.perf_counter()
    try:
        prob = model.predict_proba(X)[:, 1]
        used = "lightgbm"
    except Exception as e:
        log.error("Model scoring failed (%s). Using fallback prior.", e)
        prob = [PRIOR_CHURN_RATE] * len(X)
        used = "fallback_prior"
    elapsed = time.perf_counter() - start

    out = pd.DataFrame({
        args.id_col: ids.values,
        "churn_probability": pd.Series(prob).round(4),
        "churn_prediction": (pd.Series(prob) >= threshold).astype(int),
        "model_used": used,
    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    log.info("Scored %d rows in %.3f s (%.0f records/sec)",
             len(out), elapsed, len(out) / elapsed)
    log.info("Predicted churners: %d (%.1f%%)",
             out["churn_prediction"].sum(), 100 * out["churn_prediction"].mean())
    log.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()