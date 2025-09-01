"""
Question 6: Simple Linear Regression using Python
Write a program to implement Simple Linear Regression using python
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data: House area vs Price
np.random.seed(42)
area = np.array([1000, 1200, 1500, 1800, 2000, 2500]).reshape(-1, 1)
price = np.array([50, 55, 65, 75, 85, 100])

# Train model
model = LinearRegression()
model.fit(area, price)

# Get model parameters
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model Parameters:")
print(f"Equation: Price = {intercept:.2f} + {slope:.4f} × Area")

# Make predictions
y_pred = model.predict(area)
r2 = r2_score(price, y_pred)

print(f"R² Score: {r2:.4f}")

# Predict for new area
new_area = np.array([[1600]])
predicted_price = model.predict(new_area)[0]
print(f"Predicted price for 1600 sq ft: ${predicted_price:.2f}k")

# Visualization
plt.scatter(area, price, color='blue', label='Actual Data')
plt.plot(area, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (thousands $)')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

