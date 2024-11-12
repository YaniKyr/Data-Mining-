import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import dataArchive as dA

def plot_clusters(data, labels, title):

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.show()

def load_data():

    data = np.zeros((len(dA.csvs), len(dA.activity_labels)))
    labels = np.zeros((len(dA.csvs), len(dA.activity_labels)))
    for key, value in dA.csvs.items():
        try:
            df = pd.read_csv(value)
            lab, val = np.unique(df['label'], return_counts=True)
            for i in range(len(lab)):
                data[key-1, i] = val[i]
                labels[key-1, i] = lab[i]
        except Exception as e:
            print(f"Error loading data from {value}: {e}")
    return data, labels

def plot_elbow_method(data):

    inertias = []
    for i in range(1, data.shape[0]):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        inertias.append(km.inertia_)
    plt.plot(range(1, data.shape[0]), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def evaluate_clustering(data, labels, true_labels, algorithm_name):

    silhouette = silhouette_score(data, labels)
    ari = np.mean([adjusted_rand_score(true_labels[:, i], labels) for i in range(true_labels.shape[1])])
    nmi = np.mean([normalized_mutual_info_score(true_labels[:, i], labels) for i in range(true_labels.shape[1])])
    print(f'{algorithm_name} Silhouette Score: {silhouette}')
    print(f'{algorithm_name} Adjusted Rand Index: {ari}')
    print(f'{algorithm_name} Normalized Mutual Information: {nmi}')
    return silhouette, ari, nmi

def run_clusters(n_clusters=3):

    data, true_labels = load_data()

    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    n_clusters = 3

    # Plot elbow method
    plot_elbow_method(data)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    km_labels = kmeans.fit_predict(data)
    plot_clusters(data, km_labels, 'KMeans Clustering')
    km_silhouette, km_ari, km_nmi = evaluate_clustering(data, km_labels, true_labels, 'KMeans')

    # Agglomerative clustering
    hier_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hier_clustering.fit_predict(data)
    plot_clusters(data, hier_labels, 'Agglomerative Clustering')
    hier_silhouette, hier_ari, hier_nmi = evaluate_clustering(data, hier_labels, true_labels, 'Agglomerative Clustering')

    # Print inertia for KMeans
    km_inertia = kmeans.inertia_
    print(f'KMeans Inertia: {km_inertia}')

if __name__ == "__main__":
    run_clusters()