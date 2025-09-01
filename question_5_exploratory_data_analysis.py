"""
Question 5: Exploratory Data Analysis in Python
Write a program for Exploratory Data Analysis in Python
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load sample dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("=== EXPLORATORY DATA ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nBasic Statistics:")
print(df.describe())

print(f"\nTarget Distribution:")
print(df['species'].value_counts())

print(f"\nCorrelation Matrix:")
correlation_matrix = df[iris.feature_names].corr()
print(correlation_matrix)

# Visualizations
plt.figure(figsize=(12, 8))

# Correlation heatmap
plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

# Species distribution
plt.subplot(2, 2, 2)
df['species'].value_counts().plot(kind='bar')
plt.title('Species Distribution')

# Scatter plot by species
plt.subplot(2, 2, 3)
for species in iris.target_names:
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], 
               label=species, alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Width by Species')
plt.legend()

plt.tight_layout()
plt.show()

