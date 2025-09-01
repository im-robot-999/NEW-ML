"""
Question 4: Process of Normalizing the Range of Features in a Dataset
Write a program to process of normalising the range of features in a dataset
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Create sample dataset with different scales
data = {
    'Age': [25, 30, 35, 40, 45, 50],
    'Income': [30000, 60000, 90000, 120000, 150000, 200000],
    'Experience': [2, 5, 8, 15, 20, 25]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Method 1: Min-Max Scaling (best for neural networks)
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns)

print("\nAfter Min-Max Scaling (0-1 range):")
print(df_minmax)

# Method 2: Standard Scaling (best for SVM, KNN)
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

print("\nAfter Standard Scaling (mean=0, std=1):")
print(df_standard)

