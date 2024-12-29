import numpy as np

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q + 1e-9), axis=1)

def inception_score(predictions, eps=1e-9):
    p_y = np.mean(predictions, axis=0)
    kl_div = kl_divergence(predictions, p_y)
    mean_kl_div = np.mean(kl_div)
    return np.exp(mean_kl_div)
