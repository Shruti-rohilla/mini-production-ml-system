from data_pipeline import clean_data, load_data, split_data

df = clean_data(load_data())
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

out = X_test.copy()
out.insert(0, "CustomerID", df.loc[X_test.index, "CustomerID"])
out.to_csv("data/sample_input.csv", index=False)
print("Saved data/sample_input.csv with", len(out), "rows")