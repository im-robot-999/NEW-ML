"""
Question 10: Using the Elbow Method to Pick a Good K Value in KNN Model
Write a program to use the elbow method to pick a good K Value in the Model
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different k values
k_values = range(1, 11)
cv_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Cross-validation score
    cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5).mean()
    cv_scores.append(cv_score)
    
    # Test score
    knn.fit(X_train_scaled, y_train)
    test_score = knn.score(X_test_scaled, y_test)
    test_scores.append(test_score)

# Find elbow using second derivative
def find_elbow(scores):
    d2 = np.gradient(np.gradient(scores))
    return np.argmax(d2) + 1  # +1 because k starts from 1

optimal_k = find_elbow(cv_scores)
print(f"Optimal k (using elbow method): {optimal_k}")
print(f"CV Score at optimal k: {cv_scores[optimal_k-1]:.4f}")

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, cv_scores, 'bo-', label='CV Score')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Elbow k={optimal_k}')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Elbow Method: CV Accuracy vs k')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, test_scores, 'go-', label='Test Score')
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Elbow k={optimal_k}')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs k')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

