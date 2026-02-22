import pandas as pd

# Load dataset
df = pd.read_csv("employees.csv")

# Select specific columns
selected_columns = df[["Name", "Salary"]]
print(f"Selected Columns:\n{selected_columns}")

# Filter employees with salary greater than 60000
high_salary = df[df["Salary"] > 60000]
print(f"\nEmployees with Salary > 60000:\n{high_salary}")

# Filter employees in IT department
it_employees = df[df["Department"] == "IT"]
print(f"\nIT Department Employees:\n{it_employees}")
