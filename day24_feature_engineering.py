# Day 24 — Feature Engineering

import pandas as pd

# Sample dataset
data = {
    "Name":       ["Chukwuemeka", "Fatima", "Adaeze", "Tunde", "Ngozi", "Emeka"],
    "Department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing", "Engineering"],
    "Salary":     [85000, 62000, 91000, 74000, 58000, 95000],
    "Experience": [5, 3, 8, 4, 2, 10],
    "Age":        [28, 25, 32, 30, 24, 36],
}

df = pd.DataFrame(data)

print("Original Dataset")
print(df)

# 1. Create a new column: Salary per Year of Experience
df["Salary_Per_Year_Exp"] = (df["Salary"] / df["Experience"]).round(2)

# 2. Binning — categorise Experience into levels
bins = [0, 3, 6, 10]
labels = ["Junior", "Mid-level", "Senior"]
df["Experience_Level"] = pd.cut(df["Experience"], bins=bins, labels=labels)

# 3. Binning — categorise Age into groups
age_bins = [20, 27, 32, 40]
age_labels = ["Early Career", "Mid Career", "Experienced"]
df["Career_Stage"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

# 4. Encoding — convert Department (categorical) to numeric
df["Department_Encoded"] = df["Department"].astype("category").cat.codes

# 5. Flag — identify high earners (above median salary)
median_salary = df["Salary"].median()
df["High_Earner"] = df["Salary"].apply(lambda x: 1 if x > median_salary else 0)

print("\nEngineered Dataset ")
print(df)

print("\n Feature Summary ")
print(df[["Name", "Experience_Level", "Career_Stage", "Salary_Per_Year_Exp", "High_Earner"]])