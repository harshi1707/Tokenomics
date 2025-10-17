import numpy as np
from sklearn.cluster import KMeans


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 8):
    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = km.fit_predict(embeddings)
    return labels, km

