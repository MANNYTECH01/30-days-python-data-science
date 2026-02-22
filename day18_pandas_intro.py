import pandas as pd

# Creating a DataFrame
data = {
    "Hours Studied": [1, 2, 3, 4, 5, 6],
    "Exam Score": [50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

print("Full DataFrame:\n")
print(df)

print("\nAccessing a single column (Series):\n")
print(df["Exam Score"])