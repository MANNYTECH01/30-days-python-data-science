import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
hours = np.array([1, 2, 3, 4, 5, 6])
scores = np.array([50, 55, 65, 70, 80, 90])

plt.figure(figsize=(12, 4))

# Scatter plot – relationship
plt.subplot(1, 3, 1)
plt.scatter(hours, scores)
plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

# Bar chart – comparison
plt.subplot(1, 3, 2)
plt.bar(hours, scores)
plt.title("Score Comparison by Study Hours")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

# Histogram – distribution
plt.subplot(1, 3, 3)
plt.hist(scores, bins=5)
plt.title("Distribution of Exam Scores")
plt.xlabel("Score Range")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()