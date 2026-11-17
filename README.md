\# Mini Production ML System: Telco Customer Churn



End-to-end churn prediction system: leak-free preprocessing, baseline + LightGBM,

FastAPI online service, batch CLI scoring, latency profiling, robustness tests,

and a PSI/KS drift-monitoring simulation.



\## Setup (Windows)



```

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



Place the dataset at `data/WA\_Fn-UseC\_-Telco-Customer-Churn.csv.xlsx`.



\## Run



Train and evaluate (run from project root):



```

python src\\train.py

python src\\evaluate.py

```



Online API:



```

uvicorn src.serving.app:app --reload

```



Open http://127.0.0.1:8000/docs, or:



```

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json

```



Batch scoring:



```

python src\\make\_sample.py

python src\\predict\_batch.py --input data\\sample\_input.csv --output outputs\\predictions.csv

```



Profiling, robustness, drift, tests:



```

python src\\serving\\benchmark.py

python src\\serving\\robustness\_test.py

python src\\drift\_monitor.py

python -m pytest tests -v

```



\## Design notes



\- Stratified 60/20/20 split (dataset has no timestamp).

\- Preprocessors fitted on train only; leakage columns (Churn Label, Churn Score,

&#x20; Churn Reason, CLTV) dropped.

\- Imbalance handled with class weights; threshold tuned on validation.

\- Fallback: if model scoring fails, API returns the prior churn rate.



\## Structure



```

src/            data\_pipeline.py, train.py, evaluate.py, predict\_batch.py, drift\_monitor.py

src/serving/    app.py, schemas.py, benchmark.py, robustness\_test.py

tests/          test\_api.py, test\_fallback.py

artifacts/      serialized models and thresholds

```

