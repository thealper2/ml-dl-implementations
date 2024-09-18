import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

data = np.random.randn(100) + np.sin(np.linspace(0, 10, 100))

window_size = 5
smoothed_data = moving_average(data, window_size)

plt.figure(figsize=(8, 6))
plt.plot(data, label="Original Data", alpha=0.5)
plt.plot(range(window_size - 1, len(data)), smoothed_data, label="Moving Average", color="red")
plt.legend()
plt.savefig("moving_average.png")