import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. Load and prepare data
print("Loading data...")
data_path = os.path.join("data", "CMO-Historical-Data-Monthly-CLEANED.xlsx")
df = pd.read_excel(data_path, sheet_name="Cleaned Data", index_col="Date")

# Get last 12 months with proper date handling
recent_date = df.index[-1] - pd.DateOffset(months=11)
df_recent = df.loc[recent_date:].T
df_recent = df_recent.dropna(how='all')  # Remove empty commodities

# 2. Enhanced preprocessing
print("\nPreprocessing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_recent)

# 3. Advanced cluster evaluation
def evaluate_clusters(X, max_clusters=10):
    results = []
    for k in range(2, min(max_clusters, len(X)-1)+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        results.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X, labels),
            'calinski': calinski_harabasz_score(X, labels)
        })
    return pd.DataFrame(results)

cluster_results = evaluate_clusters(X_scaled)
print("\nCluster evaluation results:")
print(cluster_results)

# 4. Determine optimal clusters with multiple metrics
optimal_k = cluster_results.loc[
    (cluster_results['silhouette'].idxmax() + 
     cluster_results['calinski'].idxmax()) // 2, 'k'
]
optimal_k = int(optimal_k)
print(f"\nOptimal number of clusters: {optimal_k}")

# 5. Train final model with balanced clusters if needed
if optimal_k == 2 and len(df_recent) > 20:  # If potentially imbalanced
    print("\nDetecting potential cluster imbalance...")

    # Try with increased n_init and different initialization
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50, init='random')
    kmeans.fit(X_scaled)

    # Check cluster balance
    cluster_counts = pd.Series(kmeans.labels_).value_counts()
    imbalance_ratio = cluster_counts.max() / cluster_counts.min()

    if imbalance_ratio > 5:  # Significant imbalance
        print(f"High imbalance detected ({imbalance_ratio:.1f}:1). Trying alternative approaches...")

        # Option 1: Try different k
        if cluster_results['k'].max() > 2:
            optimal_k = cluster_results.loc[cluster_results['silhouette'].idxmax(), 'k']
            print(f"Trying k={optimal_k} based on silhouette score")
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
            kmeans.fit(X_scaled)

        # Option 2: Use hierarchical clustering initialization
        if (pd.Series(kmeans.labels_).value_counts().max() /
            pd.Series(kmeans.labels_).value_counts().min() > 5):
            print("Still imbalanced. Using kmeans++ initialization with more iterations")
            kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=100)
            kmeans.fit(X_scaled)
else:
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)

# 6. Analyze and visualize results
df_recent['Cluster'] = kmeans.labels_

# Cluster visualization with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.labels_, palette='Set2', s=100)
plt.title("Commodity Clusters (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.tight_layout()

# 7. Save plot and results
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

plot_path = os.path.join(data_dir, "commodity_clusters.png")
plt.savefig(plot_path)
print(f"\nCluster visualization saved to: {plot_path}")

# Save clustering model and scaler
joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "kmeans_scaler.pkl"))
print("KMeans model and scaler saved.")

# Save clustered data
clustered_df = df_recent.copy()
clustered_df.index.name = "Commodity"
clustered_df.reset_index(inplace=True)

excel_path = os.path.join(data_dir, "clustered_commodities.xlsx")
clustered_df.to_excel(excel_path, index=False)
print(f"Clustered commodity data saved to: {excel_path}")

print("\nClustering complete!")
print(f"Final number of clusters: {optimal_k}")
print(f"Clustered data shape: {clustered_df.shape}")