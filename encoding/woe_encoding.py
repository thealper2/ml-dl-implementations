import numpy as np
from collections import defaultdict

def woe_encoding(categories, target):
    category_counts = defaultdict(lambda: [0, 0])

    for category, t in zip(categories, target):
        category_counts[category][t] += 1

    woe_map = {}
    epsilon = 1e-10
    total_good = sum([t == 1 for t in target])
    total_bad = sum([t == 0 for t in target])

    for category, (good, bad) in category_counts.items():
        p_good = good / total_good
        p_bad = bad / total_bad
        woe_map[category] = np.log((p_good + 10) / (p_bad + epsilon))

    encoded_values = [woe_map[category] for category in categories]
    return encoded_values

categories = ['A', 'B', 'A', 'C', 'B', 'A', 'C']
target = [1, 0, 1, 0, 1, 0, 0]

encoded_values = woe_encoding(categories, target)

print("Weights of Evidence (WoE) Encoding:")
for i in range(len(categories)):
    print(f"{categories[i]} => {encoded_values[i]}")