import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv("D:/MSC 3/ML (1)/ML/prac2/data.csv")  # Replace "data.csv" with your dataset file
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']  # Target variable (if applicable)


pca = PCA()
pca.fit(X)
explained_var_ratio = pca.explained_variance_ratio_


plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()


num_components = 5  


pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)


loadings = pca.components_
feature_scores = abs(loadings).mean(axis=0)  


feature_scores_df = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})


feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)


print(feature_scores_df)
