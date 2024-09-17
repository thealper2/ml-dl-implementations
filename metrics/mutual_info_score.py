import numpy as np

def mutual_information(x, y, bins=10):
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)

    c_x = np.sum(c_xy, axis=1)
    c_y = np.sum(c_xy, axis=0)
    
    p_xy = c_xy / np.sum(c_xy)
    
    p_x = c_x / np.sum(c_x)
    p_y = c_y / np.sum(c_y)
    
    mi = 0.0
    for i in range(c_xy.shape[0]):
        for j in range(c_xy.shape[1]):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 1, 4, 5, 4, 7, 8, 9, 10])

mi_score = mutual_information(x, y, bins=5)
print(f"Mutual Information Score: {mi_score:.2f}")
