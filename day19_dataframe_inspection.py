import pandas as pd

# Sample employee dataset
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Experience (Years)": [2, 5, 3, 7, 4],
    "Salary": [50000, 60000, 55000, 75000, 62000]
}

df = pd.DataFrame(data)

print("First few rows:\n")
print(df.head())

print("\nDataFrame Information:\n")
df.info()

print("\nStatistical Summary:\n")
print(df.describe())