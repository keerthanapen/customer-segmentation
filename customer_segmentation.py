import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#load dataset
df = pd.read_csv('data/customers.csv')

print(df.head())
#Data Preprocessing
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
#apply KMeans cluster
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)

df['Cluster'] = kmeans.labels_

print(df.head())
#Visualize Clusters Using PCA (Reduce to 2D)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1')
plt.title('Customer Segments Visualization')
plt.show()
plt.savefig('customer_segments.png')

