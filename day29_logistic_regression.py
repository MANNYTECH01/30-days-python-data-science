# Day 29 — Classification: Logistic Regression & Confusion Matrix

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Sample dataset
data = {
    "Name":       ["Chukwuemeka", "Fatima", "Adaeze", "Tunde", "Ngozi", "Emeka",
                   "Bola", "Kelechi", "Amara", "Suleiman", "Yetunde", "Ibrahim",
                   "Chidi", "Halima", "Obinna", "Aisha"],
    "Department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing",
                   "Engineering", "HR", "Engineering", "Marketing", "HR",
                   "Engineering", "Marketing", "HR", "Engineering", "Marketing", "HR"],
    "Salary":     [85000, 62000, 91000, 74000, 58000, 95000, 70000, 88000,
                   61000, 76000, 99000, 64000, 72000, 93000, 59000, 68000],
    "Experience": [5, 3, 8, 4, 2, 10, 6, 7, 3, 5, 11, 4, 5, 9, 2, 4],
    "Age":        [28, 25, 32, 30, 24, 36, 31, 33, 26, 29, 38, 27, 30, 35, 24, 28],
}

df = pd.DataFrame(data)

# 1. Create the target variable: High Earner (1) or Not (0)
median_salary = df["Salary"].median()
df["High_Earner"] = (df["Salary"] > median_salary).astype(int)

print("- Dataset with Target Variable -")
print(df[["Name", "Salary", "High_Earner"]])

# 2. Encode categorical column
le = LabelEncoder()
df["Department_Encoded"] = le.fit_transform(df["Department"])

# 3. Define features and target
X = df[["Department_Encoded", "Experience", "Age"]]
y = df["High_Earner"]

# 4. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Make predictions
predictions = model.predict(X_test_scaled)

# 8. Evaluate performance
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"\n- Model Accuracy: {accuracy:.2%} -")

print("\n- Confusion Matrix -")
print(cm)

print("\n- Classification Report -")
print(classification_report(y_test, predictions))

# 9. Compare predictions to actual values
results = pd.DataFrame({
    "Actual":    y_test.values,
    "Predicted": predictions
})
print("\n- Predictions vs Actual -")
print(results)