"""
Question 1: Working with Missing Values in Pandas
Write a program for Working with Missing values in Pandas
"""

import pandas as pd
import numpy as np

# Create sample DataFrame with missing values
data = {
    'Name': ['John', 'Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, np.nan, 30, np.nan, 28],
    'Salary': [50000, 60000, np.nan, 70000, np.nan],
    'Department': ['IT', 'HR', 'IT', np.nan, 'Finance']
}

df = pd.DataFrame(data)
print("Original DataFrame with Missing Values:")
print(df)
print("\nMissing values count:")
print(df.isnull().sum())

# Method 1: Fill missing values with mean (for numerical columns)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Method 2: Fill missing values with mode (for categorical columns)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)

print("\nDataFrame after handling missing values:")
print(df)
print("\nMissing values count after handling:")
print(df.isnull().sum())

