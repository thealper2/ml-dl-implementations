import numpy as np
import matplotlib.pyplot as plt

def lowess(x, y, f=0.25, iters=3):
    n = len(x)
    r = int(np.ceil(f * n))
    smoothed_y = np.zeros(n)

    for i in range(n):
        distances = np.abs(x - x[i])
        sorted_indices = np.argsort(distances)
        closest_x = x[sorted_indices[:r]]
        closest_y = y[sorted_indices[:r]]

        max_distance = distances[sorted_indices[r - 1]]
        weights = (1 - (distances[sorted_indices[:r]] / max_distance) ** 3) ** 3

        b = np.polyfit(closest_x, closest_y, 1, w=weights)
        smoothed_y[i] = np.polyval(b, x[i])

    return smoothed_y

x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.2, 50)
smoothed_y = lowess(x, y, f=0.2)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Original Data", alpha=0.5)
plt.plot(x, smoothed_y, label="LOWESS Smoothed", color="red")
plt.legend()
plt.savefig("lowess.png")