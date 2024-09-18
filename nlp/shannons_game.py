import numpy as np
from scipy.special import entr

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    return np.sum(entr(probabilities).real)

def channel_capacity(bitrate, noise_power, signal_power):
    return bitrate * np.log2(1 + (signal_power / noise_power))

data = np.array([1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])

entropy = shannon_entropy(data)
print(f"Entropy: {entropy:.4f} bit")

bitrate = 1e6
noise_power = 0.1
signal_power = 1.0
capacity = channel_capacity(bitrate, noise_power, signal_power)
print(f"Channel capacity: {capacity:.2f} bps")
