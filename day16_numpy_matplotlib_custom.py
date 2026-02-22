import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 25, 30])

plt.plot(x, y, color='purple', marker='o', linestyle='--', label='Growth Trend')

plt.title("Customized Plot with NumPy Data")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()

plt.show()