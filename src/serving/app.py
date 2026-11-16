import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

sys.path.append(str(Path(__file__).resolve().parent))
from schemas import CustomerFeatures, PredictionResponse

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
PRIOR_CHURN_RATE = 0.2654  # fallback: train set ka churn rate

app = FastAPI(title="Churn Prediction API")

model = joblib.load(ARTIFACTS / "lgbm.joblib")
threshold = joblib.load(ARTIFACTS / "thresholds.joblib")["lgbm_threshold"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    row = customer.model_dump(by_alias=True)
    if row["Total Charges"] is None:
        row["Total Charges"] = 0.0
    df = pd.DataFrame([row])

    try:
        prob = float(model.predict_proba(df)[0, 1])
        used = "lightgbm"
    except Exception:
        # Fallback: model fail ho to prior churn rate
        prob = PRIOR_CHURN_RATE
        used = "fallback_prior"

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=int(prob >= threshold),
        threshold=round(threshold, 4),
        model_used=used,
    )