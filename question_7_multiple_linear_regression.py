"""
Question 7: Multiple Linear Regression using Python
Write a program to implement multiple Linear Regression using python
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample dataset: Area, Bedrooms, Age → Price
X = np.array([
    [1000, 2, 5],   # Area, Bedrooms, Age
    [1200, 2, 8],
    [1500, 3, 12],
    [1800, 3, 15],
    [2000, 4, 20]
])
y = np.array([50, 55, 65, 75, 85])  # Price in thousands

# Train model
model = LinearRegression()
model.fit(X, y)

# Get model parameters
coefficients = model.coef_
intercept = model.intercept_
feature_names = ['Area', 'Bedrooms', 'Age']

print("Model Parameters:")
print(f"Intercept: {intercept:.2f}")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

# Make predictions
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\nR² Score: {r2:.4f}")

# Predict for new house
new_house = np.array([[1600, 3, 10]])  # Area, Bedrooms, Age
predicted_price = model.predict(new_house)[0]

print(f"\nPrediction for new house:")
print(f"Area: 1600 sq ft, Bedrooms: 3, Age: 10 years")
print(f"Predicted Price: ${predicted_price:.2f}k")

