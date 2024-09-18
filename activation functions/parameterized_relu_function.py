def parameterized_relu(x, alpha=0.01):
    if x < 0:
        return alpha * x
    else:
        return x