import numpy as np

def elu(z, alpha=0.01):
    return max(alpha * (np.exp(z) - 1), z)