# ===============================
# Customer Segmentation Analysis
# ===============================
# Project: Customer Segmentation Analysis
# Algorithm: K-Means Clustering
# Language: Python
# Libraries: pandas, numpy, matplotlib, seaborn, sklearn

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
# 👉 CSV file cha full path ith change kar
df = pd.read_csv(r"D:\PROJECT\Customer Segmentation Analysis\Mall_Customers.csv")

print("Dataset Loaded Successfully ✅")
print(df.head())

# 3. Dataset Information
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 4. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

# Missing values astil tr drop / fill
df.dropna(inplace=True)

# 5. Feature Selection
print("\nColumns in Dataset:", df.columns)

features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 6. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 7. Finding Optimal K (Elbow Method)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()

# 8. Apply K-Means Clustering
# Elbow graph varun optimal K select kara, example K=4
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nCluster Distribution:")
print(df['Cluster'].value_counts())

# 9. Cluster Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1',
    s=100
)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# 10. Cluster Analysis
# Numeric columns only for mean
numeric_cols = df.select_dtypes(include=np.number).columns
cluster_summary = df.groupby('Cluster')[numeric_cols].mean()
print("\nCluster Summary (Numeric Columns):")
print(cluster_summary)

# Categorical columns: Example Gender majority per cluster
if 'Gender' in df.columns:
    cluster_gender = df.groupby('Cluster')['Gender'].agg(lambda x: x.mode()[0])
    print("\nCluster Majority Gender:")
    print(cluster_gender)

# 11. Business Insights
"""
Cluster 0 → High income, high spending (Premium Customers)
Cluster 1 → Low income, low spending (Low Value Customers)
Cluster 2 → Medium income, medium spending (Potential Loyal Customers)
Cluster 3 → High income, low spending (Cautious / Discount Seekers)
"""

print("\nInsights Generated Successfully 🎯")
