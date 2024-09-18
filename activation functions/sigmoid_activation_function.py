import math
import numpy as np

def sigmoid_math(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

z = 0
result_math = sigmoid_math(z)
result_np = sigmoid_np(z)
print(result_math) # 0.5
print(result_np) # 0.5