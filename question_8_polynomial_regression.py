"""
Question 8: Implementation of Polynomial Regression in Python
Write a program to Implementation of Polynomial Regression in python
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Generate sample data with non-linear relationship
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2

print("=== POLYNOMIAL REGRESSION ===")
print("Data: X =", X.flatten(), "Y =", y)

# Test different polynomial degrees
degrees = [1, 2, 3]
for degree in degrees:
    # Create polynomial pipeline
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit model
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"Degree {degree}: R² = {r2:.4f}")

# Best model (degree 2 for quadratic data)
best_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
best_model.fit(X, y)

# Predict for new value
X_new = np.array([[6]])
y_new = best_model.predict(X_new)
print(f"\nPrediction for X=6: Y={y_new[0]:.2f}")

# Visualization
X_plot = np.linspace(1, 6, 100).reshape(-1, 1)
y_plot = best_model.predict(X_plot)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(True)
plt.show()

