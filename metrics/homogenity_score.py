import numpy as np

def homogeneity_score(labels_true, labels_pred):
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    homogeneity_scores = []
    
    for cluster in unique_pred:
        cluster_mask = labels_pred == cluster
        cluster_size = np.sum(cluster_mask)
        class_distribution = [np.sum(labels_true[cluster_mask] == cls) / cluster_size for cls in unique_true]
        
        homogeneity_scores.append(np.max(class_distribution))
    
    homogeneity_score_value = np.mean(homogeneity_scores)
    
    return homogeneity_score_value

labels_true = np.array([0, 0, 1, 1, 2, 2, 1, 0])
labels_pred = np.array([0, 0, 1, 1, 2, 2, 2, 1])

homogeneity_score_value = homogeneity_score(labels_true, labels_pred)
print(f"Homogeneity Score: {homogeneity_score_value:.2f}")
