import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('C:/Users/kamle/Downloads/ML/prac7/abalone_csv.csv')
X = dataset.iloc[:, [3, 4]].values

# Dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distances')
plt.show()

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
print("Prediction Values: ", y_hc)

# Scatter Plot
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plot Settings
plt.title('Clusters of Abalones (Hierarchical Clustering Model)')
plt.xlabel('Feature 1')  # Update with appropriate label
plt.ylabel('Feature 2')  # Update with appropriate label
plt.legend()
plt.show()
