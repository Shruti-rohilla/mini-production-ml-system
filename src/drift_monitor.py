import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from data_pipeline import NUMERIC_COLS, clean_data, load_data, split_data


def psi_numeric(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def status(psi):
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "watch"
    return "ALERT"


def drift_table(ref, cur):
    rows = []
    for col in ref.columns:
        if col in NUMERIC_COLS:
            p = psi_numeric(ref[col], cur[col])
            ks = ks_2samp(ref[col], cur[col])
            rows.append((col, "numeric", p, ks.statistic, ks.pvalue, status(p)))
        else:
            p = psi_categorical(ref[col], cur[col])
            rows.append((col, "categorical", p, np.nan, np.nan, status(p)))
    return pd.DataFrame(rows, columns=["feature", "type", "PSI", "KS_stat", "KS_p", "status"])


def simulate_drift(df, seed=0):
    """Naye customers, mehnge plans, zyada month-to-month, kam tenure."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["Monthly Charges"] = d["Monthly Charges"] * 1.3
    d["Tenure Months"] = (d["Tenure Months"] * 0.5).round().astype(int)
    d["Total Charges"] = d["Monthly Charges"] * d["Tenure Months"]
    flip = rng.random(len(d)) < 0.4
    d.loc[flip, "Contract"] = "Month-to-month"
    return d


if __name__ == "__main__":
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pd.set_option("display.width", 140)
    fmt = {"PSI": "{:.3f}".format, "KS_stat": "{:.3f}".format, "KS_p": "{:.4f}".format}

    print("=== Reference (train) vs Normal production (test) ===")
    normal = drift_table(X_train, X_test)
    print(normal.to_string(index=False, formatters=fmt))
    print("\nMax PSI:", round(normal["PSI"].max(), 3))

    print("\n=== Reference (train) vs Simulated drifted production ===")
    drifted = drift_table(X_train, simulate_drift(X_test))
    print(drifted.to_string(index=False, formatters=fmt))
    print("\nMax PSI:", round(drifted["PSI"].max(), 3))
    print("Features in ALERT:", list(drifted.loc[drifted["status"] == "ALERT", "feature"]))