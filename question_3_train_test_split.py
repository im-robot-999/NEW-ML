"""
Question 3: Train and Test Data Set Split Using Sklearn In Python
Write a program for train and Test data set Split Using Sklearn In Python
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, random_state=42)

print("Dataset shape:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# Method: Using train_test_split with stratification for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% for testing, 80% for training
    random_state=42,         # For reproducible results
    stratify=y               # Maintain class distribution in both sets
)

print("\nAfter splitting:")
print(f"Training set: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Verify class distribution is maintained
print("\nClass distribution:")
print(f"Original dataset: {np.bincount(y)}")
print(f"Training set: {np.bincount(y_train)}")
print(f"Testing set: {np.bincount(y_test)}")

# Calculate split percentages
train_percent = (len(X_train) / len(X)) * 100
test_percent = (len(X_test) / len(X)) * 100

print(f"\nSplit percentages:")
print(f"Training: {train_percent:.1f}%")
print(f"Testing: {test_percent:.1f}%")

