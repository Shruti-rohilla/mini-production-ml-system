import pandas as pd

df = pd.read_excel("data/WA_Fn-UseC_-Telco-Customer-Churn.csv.xlsx")

print("Shape (rows, columns):", df.shape)
print()
print(df.head())
print()
print(df.dtypes)
print()
print("Churn counts:")
print(df["Churn Value"].value_counts())
print()
print("Churn percentage:")
print(df["Churn Value"].value_counts(normalize=True) * 100)
print()
print("Total Charges ke andar blank values:")
print(df["Total Charges"].apply(lambda x: str(x).strip() == "").sum())
