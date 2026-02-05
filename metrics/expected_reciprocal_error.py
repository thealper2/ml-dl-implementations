def err_at_k(relevances, k, rel_max=None):
    if rel_max is None:
        rel_max = max(relevances) if relevances else 0

    err = 0.0
    p_continue = 1.0
    for i, rel in enumerate(relevances[:k], start=1):
        R_i = (2**rel - 1) / (2**rel_max)
        err += p_continue * R_i / i
        p_continue *= (1 - R_i)

    return err
