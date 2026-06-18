# Day 28 — Linear Regression: Training & Evaluation

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sample dataset
data = {
    "Name":       ["Chukwuemeka", "Fatima", "Adaeze", "Tunde", "Ngozi", "Emeka",
                   "Bola", "Kelechi", "Amara", "Suleiman", "Yetunde", "Ibrahim"],
    "Department": ["Engineering", "Marketing", "Engineering", "HR", "Marketing",
                   "Engineering", "HR", "Engineering", "Marketing", "HR",
                   "Engineering", "Marketing"],
    "Salary":     [85000, 62000, 91000, 74000, 58000, 95000, 70000, 88000,
                   61000, 76000, 99000, 64000],
    "Experience": [5, 3, 8, 4, 2, 10, 6, 7, 3, 5, 11, 4],
    "Age":        [28, 25, 32, 30, 24, 36, 31, 33, 26, 29, 38, 27],
}

df = pd.DataFrame(data)

# 1. Encode categorical column
le = LabelEncoder()
df["Department_Encoded"] = le.fit_transform(df["Department"])

# 2. Define features and target
X = df[["Department_Encoded", "Experience", "Age"]]
y = df["Salary"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Make predictions
predictions = model.predict(X_test_scaled)

# 7. Evaluate model performance
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("- Model Evaluation -")
print(f"Mean Absolute Error (MAE):  {mae:.2f}")
print(f"Root Mean Squared Error:    {rmse:.2f}")
print(f"R-squared (R²):             {r2:.4f}")

# 8. Compare predictions to actual values
results = pd.DataFrame({
    "Actual Salary":    y_test.values,
    "Predicted Salary": predictions.round(2)
})
print("\n- Predictions vs Actual -")
print(results)

# 9. Inspect model coefficients
print("\n- Feature Coefficients -")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")