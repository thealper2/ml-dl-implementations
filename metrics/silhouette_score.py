from scipy.spatial.distance import pdist, squareform

def silhouette_score(X, labels):
    n = len(X)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    distance_matrix = squareform(pdist(X, metric='euclidean'))
    silhouette_values = np.zeros(n)
    for i in range(n):
        cluster_i = labels[i]
        same_cluster_indices = np.where(labels == cluster_i)[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != i]
        if len(same_cluster_indices) > 0:
            a_i = np.mean(distance_matrix[i, same_cluster_indices])
        else:
            a_i = 0
        
        b_i = float('inf')
        for cluster_j in unique_labels:
            if cluster_j == cluster_i:
                continue
                
            other_cluster_indices = np.where(labels == cluster_j)[0]
            mean_distance = np.mean(distance_matrix[i, other_cluster_indices])
            b_i = min(b_i, mean_distance)
        
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0
    
    return np.mean(silhouette_values)
