# Day 30 — End-to-End Mini Project: Titanic Survival Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("data/titanic.csv")

print("--- Dataset Overview ---")
print(df.shape)
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 2. Data cleaning
# Drop columns that carry little predictive value or are mostly missing
df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 3. Feature engineering
# Combine SibSp and Parch into a single family size feature
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

# Flag passengers travelling alone
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# 4. Encode categorical columns
le_sex = LabelEncoder()
df["Sex_Encoded"] = le_sex.fit_transform(df["Sex"])

le_embarked = LabelEncoder()
df["Embarked_Encoded"] = le_embarked.fit_transform(df["Embarked"])

# 5. Exploratory analysis
print("\n--- Survival Rate by Sex ---")
print(df.groupby("Sex")["Survived"].mean().round(2))

print("\n--- Survival Rate by Passenger Class ---")
print(df.groupby("Pclass")["Survived"].mean().round(2))

print("\n--- Survival Rate by Family Size ---")
print(df.groupby("Family_Size")["Survived"].mean().round(2))

# 6. Define features and target
features = ["Pclass", "Sex_Encoded", "Age", "Fare",
            "Family_Size", "Is_Alone", "Embarked_Encoded"]

X = df[features]
y = df["Survived"]

# 7. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 10. Make predictions
predictions = model.predict(X_test_scaled)

# 11. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- Model Accuracy: {accuracy:.2%} ---")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# 12. Inspect feature influence
print("\n--- Feature Coefficients ---")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")