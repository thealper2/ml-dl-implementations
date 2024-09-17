import numpy as np

def one_hot_encoder(x):
    arr = []
    for val in x:
        data = [0] * (max(x) + 1)
        data[val] = 1
        arr.append(data)

    return arr

x = np.array([0, 1, 2, 1, 0, 5, 0, 3, 4, 4, 3, 1, 3])
ohe = one_hot_encoder(x)
print("OHE:")
for i in range(len(x)):
    print(f"{x[i]} => {ohe[i]}")