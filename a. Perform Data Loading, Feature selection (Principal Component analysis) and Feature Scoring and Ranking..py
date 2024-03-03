import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data Loading
data = pd.read_csv("D:/MSC 3/ML (1)/ML/prac2/data.csv")  # Replace "data.csv" with your dataset file
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']  # Target variable (if applicable)

# Feature Selection using PCA
pca = PCA()
pca.fit(X)
explained_var_ratio = pca.explained_variance_ratio_

# Plot the explained variance to decide on the number of components to keep
plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()

# Choose the number of components that explain most of the variance
num_components = 5  # Adjust as needed

# Transform the data with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)

# Feature Scoring and Ranking
loadings = pca.components_
feature_scores = abs(loadings).mean(axis=0)  # Use mean absolute loading values

# Create a DataFrame to display feature scores
feature_scores_df = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})

# Sort features by score in descending order
feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

# Display the ranked features
print(feature_scores_df)
