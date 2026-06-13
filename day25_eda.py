# Day 25 — Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "Name":       ["Chukwuemeka", "Fatima", "Adaeze", "Tunde", "Ngozi", "Emeka",
                   "Bola", "Kelechi", "Amara", "Suleiman"],
    "Department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing",
                   "Engineering", "HR", "Engineering", "Marketing", "HR"],
    "Salary":     [85000, 62000, 91000, 74000, 58000, 95000, 70000, 88000, 61000, 76000],
    "Experience": [5, 3, 8, 4, 2, 10, 6, 7, 3, 5],
    "Age":        [28, 25, 32, 30, 24, 36, 31, 33, 26, 29],
}

df = pd.DataFrame(data)

# 1. Basic overview
print(" Dataset Shape ")
print(df.shape)

print("\n Data Types ")
print(df.dtypes)

print("\n Statistical Summary ")
print(df.describe())

# 2. Distribution of Salary
print("\n Salary Distribution ")
print(df["Salary"].value_counts().sort_index())

# 3. Average Salary by Department
print("\n Average Salary by Department ")
dept_avg = df.groupby("Department")["Salary"].mean().round(2)
print(dept_avg)

# 4. Correlation matrix (numerical columns only)
print("\n--- Correlation Matrix ---")
print(df[["Salary", "Experience", "Age"]].corr().round(2))

# 5. Visualise average salary by department
dept_avg.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_salary_by_department.png", dpi=150)
plt.show()

# 6. Salary vs Experience scatter plot
plt.scatter(df["Experience"], df["Salary"], color="darkorange", edgecolors="black")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("eda_salary_vs_experience.png", dpi=150)
plt.show()