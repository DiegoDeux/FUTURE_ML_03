import pandas as pd

df = pd.read_csv("../data/monster_com-job_sample.csv", on_bad_lines="skip", engine="python")

print("Shape (rows, columns):", df.shape)
print()
print("Columns:", df.columns.tolist())
print()
print("Top sectors:")
print(df["sector"].value_counts().head(15))