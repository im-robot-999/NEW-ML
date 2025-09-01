"""
Question 2: Handling Machine Learning Categorical Data
Write a program for Handling Machine Learning Categorical Data
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# Create sample DataFrame with categorical data
data = {
    'Name': ['John', 'Alice', 'Bob', 'Charlie', 'Diana'],
    'City': ['New York', 'London', 'Paris', 'New York', 'Tokyo'],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'Salary': [50000, 60000, 70000, 55000, 65000]
}

df = pd.DataFrame(data)
print("Original DataFrame with Categorical Data:")
print(df)

# Method 1: Label Encoding (for ordinal categorical data)
label_encoder = LabelEncoder()
df['City_Label'] = label_encoder.fit_transform(df['City'])
df['Education_Label'] = label_encoder.fit_transform(df['Education'])

print("\nDataFrame after Label Encoding:")
print(df)

# Method 2: One-Hot Encoding (for nominal categorical data)
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
city_encoded = onehot_encoder.fit_transform(df[['City']])
education_encoded = onehot_encoder.fit_transform(df[['Education']])

# Create column names for one-hot encoded features
city_columns = [f'City_{city}' for city in onehot_encoder.categories_[0][1:]]
education_columns = [f'Education_{edu}' for edu in onehot_encoder.categories_[0][1:]]

# Add one-hot encoded columns to DataFrame
for i, col in enumerate(city_columns):
    df[col] = city_encoded[:, i]

for i, col in enumerate(education_columns):
    df[col] = education_encoded[:, i]

print("\nDataFrame after One-Hot Encoding:")
print(df)

