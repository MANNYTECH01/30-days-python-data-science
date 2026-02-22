import pandas as pd

# Load dataset
df = pd.read_csv("data/employees.csv")

# Group by Department and calculate average salary
avg_salary = df.groupby("Department")["Salary"].mean()

print("Average Salary by Department:\n")
print(avg_salary)

# Group by Department and count number of employees
employee_count = df.groupby("Department")["EmployeeID"].count()

print("\nNumber of Employees per Department:\n")
print(employee_count)