scores = []

number_of_students = int(input("Enter number of students: "))

for i in range(number_of_students):
    score = int(input(f"Enter score for student {i + 1}: "))
    scores.append(score)

print("\nResults:")

for i, score in enumerate(scores, start=1):
    if score >= 70:
        remark = "Pass"
    else:
        remark = "Fail"

    print(f"Student {i}: {score} - {remark}")