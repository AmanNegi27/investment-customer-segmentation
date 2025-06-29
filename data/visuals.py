import numpy as np
# Workarounds for deprecated NumPy type aliases
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import streamlit as st

# --------------------------------------
# 1. PCA Scatter Plot for Cluster Visualization
# --------------------------------------
def plot_clusters(scaled_data, labels, title, use_streamlit=True):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    var_ratio = pca.explained_variance_ratio_
    df_plot = pd.DataFrame({
        'PC1': reduced[:, 0],
        'PC2': reduced[:, 1],
        'Cluster': labels
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=70, ax=ax, legend='full')
    ax.set_title(f"{title} (Explained Variance: {sum(var_ratio):.2%})")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title='Cluster')
    ax.grid(True, linestyle='--', alpha=0.7)

    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 2. Cluster Distribution Bar Plot
# --------------------------------------
def plot_cluster_distribution(df, cluster_col, title, use_streamlit=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x=cluster_col, palette='pastel', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.grid(True, linestyle='--', alpha=0.7)

    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 3. Risk vs ROI Scatter Plot by Cluster
# --------------------------------------
def plot_risk_roi(df, cluster_col, use_streamlit=True):
    if 'risk_score' not in df.columns or 'roi' not in df.columns:
        raise ValueError("DataFrame must contain 'risk_score' and 'roi' columns")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="risk_score",
        y="roi",
        hue=cluster_col,
        palette='Set2',
        s=70,
        ax=ax,
        legend='full'
    )
    ax.set_title("Risk Score vs ROI by Cluster")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Estimated ROI (%)")
    ax.legend(title='Cluster')
    ax.grid(True, linestyle='--', alpha=0.7)

    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 4. Box Plot for Numerical Features by Cluster
# --------------------------------------
def plot_box_by_cluster(df, feature, cluster_col, title, use_streamlit=True):
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x=cluster_col, y=feature, palette='Set2', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Cluster')
    ax.set_ylabel(feature.capitalize())
    ax.grid(True, linestyle='--', alpha=0.7)

    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 5. Stacked Bar Plot for Categorical Features by Cluster
# --------------------------------------
def plot_categorical_by_cluster(df, cat_col, cluster_col, title, use_streamlit=True):
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found in DataFrame")
    crosstab = pd.crosstab(df[cluster_col], df[cat_col], normalize='index')
    fig, ax = plt.subplots(figsize=(10, 6))
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title(title)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Proportion')
    ax.legend(title=cat_col.capitalize())
    ax.grid(True, linestyle='--', alpha=0.7)

    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 6. Pair Plot for Key Features by Cluster
# --------------------------------------
def plot_pair_by_cluster(df, features, cluster_col, title, use_streamlit=True):
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
    sns.pairplot(df, vars=features, hue=cluster_col, palette='Set2', diag_kind='kde')
    plt.suptitle(title, y=1.02)
    if use_streamlit:
        st.pyplot(plt)
    else:
        plt.show()

# --------------------------------------
# 7. Correlation Heatmap for Numerical Features
# --------------------------------------
def plot_correlation_heatmap(df, features, title, use_streamlit=True):
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 8. Elbow Curve for Optimal Cluster Count
# --------------------------------------
def plot_elbow_curve(scaled_data, max_k=10, title='Elbow Curve for KMeans', use_streamlit=True):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, max_k + 1), inertias, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.grid(True, linestyle='--', alpha=0.7)
    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# --------------------------------------
# 9. Silhouette Plot for Cluster Quality
# --------------------------------------
def plot_silhouette(scaled_data, labels, cluster_col, title, use_streamlit=True):
    silhouette_vals = silhouette_samples(scaled_data, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower, y_upper = 0, 0
    for i in sorted(set(labels)):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()
        y_upper += len(cluster_vals)
        ax.fill_betweenx(range(y_lower, y_upper), 0, cluster_vals, alpha=0.7, label=f'Cluster {i}')
        y_lower += len(cluster_vals)
    ax.axvline(x=silhouette_score(scaled_data, labels), color='red', linestyle='--', label='Avg Silhouette')
    ax.set_title(title)
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    if use_streamlit:
        st.pyplot(fig)
    else:
        plt.show()