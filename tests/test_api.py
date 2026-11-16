import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "serving"))

from app import app  # noqa: E402

client = TestClient(app)

VALID = {
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure Months": 2, "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
    "Streaming Movies": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
    "Payment Method": "Electronic check", "Monthly Charges": 85.5, "Total Charges": 171.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["model_used"] == "lightgbm"


def test_missing_total_charges_ok():
    payload = {k: v for k, v in VALID.items() if k != "Total Charges"}
    assert client.post("/predict", json=payload).status_code == 200


def test_unseen_category_does_not_crash():
    payload = {**VALID, "Contract": "Lifetime"}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("Tenure Months", "abc"),
    ("Tenure Months", 500),
    ("Monthly Charges", -50),
    ("Gender", "Unknown"),
])
def test_invalid_input_rejected(field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_empty_payload_rejected():
    assert client.post("/predict", json={}).status_code == 422