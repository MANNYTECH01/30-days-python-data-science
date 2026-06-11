# Day 23 — Data Cleaning & Handling Missing Values

import pandas as pd
import numpy as np

# Simulating a messy real-world dataset
data = {
    "Name":       ["Chukwuemeka", "Fatima", None, "Adaeze", "Tunde", "Ngozi"],
    "Department": ["Engineering", "Marketing", "Engineering", None, "HR", "Marketing"],
    "Salary":     [85000, 62000, 91000, 74000, None, 58000],
    "Experience": [5, 3, 8, None, 2, None],
}

df = pd.DataFrame(data)

print("Raw Dataset ")
print(df)

# 1. Detect missing values
print("\n Missing Value Count ")
print(df.isnull().sum())

print("\nMissing Value Percentage ")
print((df.isnull().sum() / len(df) * 100).round(2))

# 2. Drop rows where Name is missing (critical identifier)
df_cleaned = df.dropna(subset=["Name"])

# 3. Fill missing Department with the mode
dept_mode = df_cleaned["Department"].mode()[0]
df_cleaned["Department"] = df_cleaned["Department"].fillna(dept_mode)

# 4. Fill missing Salary with the median
salary_median = df_cleaned["Salary"].median()
df_cleaned["Salary"] = df_cleaned["Salary"].fillna(salary_median)

# 5. Fill missing Experience with the mean
exp_mean = df_cleaned["Experience"].mean()
df_cleaned["Experience"] = df_cleaned["Experience"].fillna(round(exp_mean, 1))

print("\nCleaned Dataset ")
print(df_cleaned)

print("\nFinal Missing Value Check ")
print(df_cleaned.isnull().sum())

# 6. Check for duplicate rows
print(f"\nDuplicate rows found: {df_cleaned.duplicated().sum()}")