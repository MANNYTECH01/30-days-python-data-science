# Day 27 — Introduction to Scikit-learn & Preprocessing Pipelines

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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

# 1. Encode categorical column
le = LabelEncoder()
df["Department_Encoded"] = le.fit_transform(df["Department"])

print(" Dataset with Encoded Department ")
print(df[["Name", "Department", "Department_Encoded", "Experience", "Age", "Salary"]])

# 2. Define features and target
X = df[["Department_Encoded", "Experience", "Age"]]
y = df["Salary"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

# 4. Build a preprocessing + model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression())
])

# 5. Train the pipeline
pipeline.fit(X_train, y_train)

# 6. Evaluate on test set
score = pipeline.score(X_test, y_test)
print(f"\nPipeline R² Score on Test Set: {round(score, 4)}")

# 7. Predict on test set
predictions = pipeline.predict(X_test)
print("\n--- Predictions vs Actual ---")
results = pd.DataFrame({
    "Actual Salary":    y_test.values,
    "Predicted Salary": predictions.round(2)
})
print(results)