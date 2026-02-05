def rbp_at_k(relevances, p=0.8, k=None):
    if k is None:
        k = len(relevances)

    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel * (p ** i)

    return (1 - p) * score
