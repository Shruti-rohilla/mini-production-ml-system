import time

import numpy as np
import requests

URL = "http://127.0.0.1:8000/predict"
N_REQUESTS = 300

PAYLOAD = {
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure Months": 2, "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
    "Streaming Movies": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
    "Payment Method": "Electronic check", "Monthly Charges": 85.5, "Total Charges": 171.0,
}

if __name__ == "__main__":
    session = requests.Session()

    # Warm-up (pehli kuch requests slow hoti hain, unko count nahi karte)
    for _ in range(20):
        session.post(URL, json=PAYLOAD)

    latencies, errors = [], 0
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        r = session.post(URL, json=PAYLOAD)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        if r.status_code != 200:
            errors += 1

    lat = np.array(latencies)
    total_s = lat.sum() / 1000
    print(f"Requests        : {N_REQUESTS}")
    print(f"Errors          : {errors}")
    print(f"p50 latency     : {np.percentile(lat, 50):.2f} ms")
    print(f"p95 latency     : {np.percentile(lat, 95):.2f} ms")
    print(f"p99 latency     : {np.percentile(lat, 99):.2f} ms")
    print(f"Mean latency    : {lat.mean():.2f} ms")
    print(f"Throughput      : {N_REQUESTS / total_s:.1f} req/sec")