import numpy as np

def v_measure_score(labels_true, labels_pred):
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    def _entropy(labels):
        labels, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    homogeneity_scores = []
    for cluster in unique_pred:
        cluster_mask = labels_pred == cluster
        cluster_size = np.sum(cluster_mask)
        class_distribution = [np.sum(labels_true[cluster_mask] == cls) / cluster_size for cls in unique_true]
        homogeneity_scores.append(np.max(class_distribution))
    homogeneity_score = np.mean(homogeneity_scores)
    
    completeness_scores = []
    for true_class in unique_true:
        true_mask = labels_true == true_class
        true_class_size = np.sum(true_mask)
        predicted_clusters = np.unique(labels_pred[true_mask])
        cluster_scores = [np.sum((labels_pred == cluster) & true_mask) / true_class_size for cluster in predicted_clusters]
        completeness_scores.append(np.max(cluster_scores))
    completeness_score = np.mean(completeness_scores)
    
    v_measure_score = (homogeneity_score * completeness_score) / (homogeneity_score + completeness_score)
    return v_measure_score

labels_true = np.array([0, 0, 1, 1, 2, 2, 1, 0])
labels_pred = np.array([0, 0, 1, 1, 2, 2, 2, 1])

v_measure_score_value = v_measure_score(labels_true, labels_pred)
print(f"V-Measure Score: {v_measure_score_value:.2f}")
