import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('ADD YOUR FILE')

df = pd.DataFrame(data)
df_cleaned = df.drop(columns=['CustomerID'])

df_cleaned = pd.get_dummies(df_cleaned, drop_first=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned)


print("Scaled data shape:", scaled_data.shape)

def find_optimal_clusters(data, max_k=10):
    wcss = []  
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

find_optimal_clusters(scaled_data)

optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

labels = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(8, 5))
scatter = sns.scatterplot(
    x=scaled_data[:, 0],
    y=scaled_data[:, 1],
    hue=labels,
    palette='viridis',
    s=50,
    legend='full' 
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    marker='X',
    label='Centroids', 
)

plt.title('Customer Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster')
plt.show()
