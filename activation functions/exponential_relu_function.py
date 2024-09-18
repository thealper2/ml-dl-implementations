import numpy as np

def exponential_relu(x, alpha):
    if x < 0:
        return alpha * (np.exp(x) - 1)
    else:
        return x