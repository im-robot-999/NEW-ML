"""
Question 14: Custom Dataset with make_blobs, Plot, and Initialize Random Centroids
Write a program for custom dataset with make_blobs, plot it and initialize the random centroids
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)

# Initialize random centroids
np.random.seed(42)
initial_centroids = X[np.random.choice(X.shape[0], 4, replace=False)]
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', marker='x', s=100, label='Initial Centroids')
plt.title('Initial Random Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

print("Initial random centroids:\n", initial_centroids)
