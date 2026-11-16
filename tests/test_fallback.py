import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "serving"))

import app as app_module  # noqa: E402

client = TestClient(app_module.app)

VALID = {
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure Months": 2, "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
    "Streaming Movies": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
    "Payment Method": "Electronic check", "Monthly Charges": 85.5, "Total Charges": 171.0,
}


class BrokenModel:
    def predict_proba(self, X):
        raise RuntimeError("simulated model failure")


def test_fallback_when_model_fails(monkeypatch):
    monkeypatch.setattr(app_module, "model", BrokenModel())
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "fallback_prior"
    assert body["churn_probability"] == round(app_module.PRIOR_CHURN_RATE, 4)