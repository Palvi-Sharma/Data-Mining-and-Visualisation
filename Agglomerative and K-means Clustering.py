
import pandas as pd, numpy as np
df = pd.read_csv("cereals (1).CSV")

df.info()
df.isnull().sum()

df = df.dropna()
df.isnull().sum()

X = df[['Calories', 'Protein','Fat','Sodium','Fiber','Carbo','Sugars','Potass','Vitamins']]

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_std= scaler.fit_transform(X)

############################## Agglomerative Clustering ###################################

# Create two clusters using complete linkage
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', metric ='euclidean') #IF Affinity is not working use metric  
cluster_labels = cluster.fit_predict(X_std)


# Create dendogram using scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Perform PCA to reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
cereals_pca = pca.fit_transform(X_std)

# Visualize Agglomerative Clustering using a dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Agglomerative Clustering")
linkage_matrix = linkage(X_std, method='complete', metric ='euclidean')
dendrogram(linkage_matrix)
plt.xlabel('Cereal Index')
plt.ylabel('Distance')
plt.show()

# Count cereals in each cluster for agglomerative clustering
cluster_counts = pd.Series(cluster_labels).value_counts()
print("Agglomerative Clustering Results (2 clusters):")
print(f"Cluster 1: {cluster_counts[0]} cereals")
print(f"Cluster 2: {cluster_counts[1]} cereals")

# Output
# Agglomerative Clustering Results (2 clusters):
# Cluster 1: 71 cereals
# Cluster 2: 3 cereals

##################################### K-means #####################################################
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=2, random_state=0)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

# Scatter plot for K-means Clustering with centroids
plt.figure(figsize=(8, 6))
plt.title("K-means Clustering (k=2)")
sns.scatterplot(x=cereals_pca[:, 0], y=cereals_pca[:, 1], hue=labels, palette="coolwarm", s=100)
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='black', marker='X', label="Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

kmeans_cluster_counts = pd.Series(labels).value_counts()
print("\nK-means Clustering Results (k=2):")
print(f"Cluster 1: {kmeans_cluster_counts[1]} cereals")
print(f"Cluster 2: {kmeans_cluster_counts[0]} cereals")

# Output
# K-means Clustering Results (k=2):
# Cluster 1: 51 cereals
# Cluster 2: 23 cereals

############################ Characteristic of Cereals ###########################################

# Task 3
# Calculate percentage differences between clusters for each nutrient
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_characteristics = pd.DataFrame(cluster_centers_original, columns=X.columns[:cluster_centers_original.shape[1]])
cluster_characteristics.index = ['Cluster 1', 'Cluster 2']

# Display cluster characteristics in a clear format
pd.set_option('display.max_columns', None)
print("Cluster Characteristics Comparison:")
print(cluster_characteristics)

# Output
# Cluster Characteristics Comparison:
#             Calories   Protein       Fat      Sodium     Fiber      Carbo    Sugars      Potass   Vitamins
# Cluster 1  114.347826  3.434783  1.782609  166.086957  4.260870  12.956522  8.043478  175.652174  27.173913
# Cluster 2  103.725490  2.098039  0.647059  160.686275  1.235294  15.529412  6.686275   63.725490  29.901961

# Distribution plot of the clusters
X['Cluster'] = labels
plt.figure(figsize=(6, 4))
sns.countplot(x=X["Cluster"])
plt.title("Distribution of Clusters")
plt.show()

centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='black', marker='X', label="Centroids")
