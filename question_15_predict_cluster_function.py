"""
Question 15: Function to Predict the Cluster for Data Points
Write a program to create the function to Predict the cluster for the datapoints
"""

from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Train KMeans
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

# Function to predict cluster for new data points
def predict_cluster(data_point):
    return kmeans.predict([data_point])[0]

# Example usage
new_point = [3, 2]
cluster = predict_cluster(new_point)
print(f"Data point {new_point} is in cluster {cluster}")
