import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def swish(x, b=1):
    return x * sigmoid(b * x)