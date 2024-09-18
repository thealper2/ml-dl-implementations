def leaky_relu(z, alpha=0.01):
    if z >= 0:
        return z:
    else:
        return alpha * z

print(leaky_relu(0))
print(leaky_relu(1))
print(leaky_relu(2, alpha=0.1))