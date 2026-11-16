import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv.xlsx"

TARGET = "Churn Value"

# Ye columns drop honge: leakage ya bekaar
DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason", "CLTV",
]

NUMERIC_COLS = ["Tenure Months", "Monthly Charges", "Total Charges"]


def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    return df


def clean_data(df):
    df = df.copy()
    # Blank Total Charges ko number banao, blank ko 0 (naye customers)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
    return df


def split_data(df, seed=42):
    """60% train, 20% validation, 20% test, stratified (churn ratio same rahe)."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + DROP_COLS)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = clean_data(load_data())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("Train:", X_train.shape, "churn rate:", round(y_train.mean(), 4))
    print("Val:  ", X_val.shape, "churn rate:", round(y_val.mean(), 4))
    print("Test: ", X_test.shape, "churn rate:", round(y_test.mean(), 4))
    print()
    print("Features:", list(X_train.columns))