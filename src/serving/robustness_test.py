import copy

import requests

URL = "http://127.0.0.1:8000/predict"

BASE = {
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure Months": 2, "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
    "Streaming Movies": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
    "Payment Method": "Electronic check", "Monthly Charges": 85.5, "Total Charges": 171.0,
}


def make(**changes):
    p = copy.deepcopy(BASE)
    p.update(changes)
    return p


def without(key):
    p = copy.deepcopy(BASE)
    del p[key]
    return p


TESTS = [
    ("Valid customer", BASE),
    ("Missing Total Charges (optional)", without("Total Charges")),
    ("Total Charges = null", make(**{"Total Charges": None})),
    ("Unseen category (Contract)", make(Contract="Lifetime")),
    ("Extreme Monthly Charges (1e6)", make(**{"Monthly Charges": 1_000_000})),
    ("Tenure = 500 (out of range)", make(**{"Tenure Months": 500})),
    ("Tenure = 'abc' (wrong type)", make(**{"Tenure Months": "abc"})),
    ("Negative Monthly Charges", make(**{"Monthly Charges": -50})),
    ("Missing required field (Contract)", without("Contract")),
    ("Invalid Gender value", make(Gender="Unknown")),
    ("Empty payload", {}),
]

if __name__ == "__main__":
    print(f"{'Test':<38} {'Status':<7} Result")
    print("-" * 95)
    for name, payload in TESTS:
        r = requests.post(URL, json=payload)
        body = r.json()
        if r.status_code == 200:
            result = f"prob={body['churn_probability']} model={body['model_used']}"
        else:
            result = str(body["detail"][0]["msg"])[:55]
        print(f"{name:<38} {r.status_code:<7} {result}")