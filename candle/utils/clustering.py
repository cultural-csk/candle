import numpy as np
from sklearn.cluster import AgglomerativeClustering


def hac_clustering(real_items, embeddings, threshold):
    if len(real_items) == 0:
        return []
    if len(real_items) == 1:
        return [real_items]

    # Normalize the embeddings to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Perform clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=threshold)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    id2cluster = {}
    for item_idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in id2cluster:
            id2cluster[cluster_id] = []
        id2cluster[cluster_id].append(real_items[item_idx])

    return sorted(id2cluster.values(), key=lambda c: len(c), reverse=True)
