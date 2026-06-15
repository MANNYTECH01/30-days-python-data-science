# Day 26 — Statistical Visualization with Seaborn

import pandas as pd
import seaborn as sns
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

sns.set_theme(style="whitegrid")

# 1. Bar plot — average salary by department
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Department", y="Salary", estimator="mean", ci=None, palette="Blues_d")
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.tight_layout()
plt.savefig("seaborn_barplot.png", dpi=150)
plt.show()

# 2. Box plot — salary distribution per department
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Department", y="Salary", palette="Set2")
plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("seaborn_boxplot.png", dpi=150)
plt.show()

# 3. Scatter plot — salary vs experience with department colour coding
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Experience", y="Salary", hue="Department", s=100, palette="deep")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("seaborn_scatterplot.png", dpi=150)
plt.show()

# 4. Heatmap — correlation matrix
plt.figure(figsize=(6, 4))
corr = df[["Salary", "Experience", "Age"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("seaborn_heatmap.png", dpi=150)
plt.show()

# 5. Histogram — age distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Age"], bins=5, kde=True, color="steelblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("seaborn_histplot.png", dpi=150)
plt.show()