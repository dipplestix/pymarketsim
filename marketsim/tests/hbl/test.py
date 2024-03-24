import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x, a, b, c):
    return 1 / (1 + np.exp(-a * (x - b))) + c

# Sample data
x_data = np.array([-2, -1, 0, 1, 2])
y_data = np.array([0.1, 0.15, 0.6, 0.9, 0.9])

# Fit the data to the sigmoid function
popt, pcov = curve_fit(sigmoid, x_data, y_data)

# Generate points for plotting the fitted sigmoid curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = sigmoid(x_fit, *popt)

print(popt, pcov)
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'bo', label='Original data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted sigmoid curve')
plt.title('Interpolation using Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.legend()
plt.grid(True)
plt.show()