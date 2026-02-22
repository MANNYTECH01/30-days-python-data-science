import pandas as pd

# Reading data from CSV file
df = pd.read_csv("employees.csv")

print("First 5 rows:\n")
print(df.head())

print(f"\nLast 5 rows:\n{df.tail()}")

print(f"\nData Information:\n{df.info()}")

print(f"\nStatistical Summary:\n{df.describe()}")
