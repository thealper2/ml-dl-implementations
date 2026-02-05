def kendall_tau_distance(rank1, rank2):
    assert len(rank1) == len(rank2)
    
    n = len(rank1)
    pos2 = {item: idx for idx, item in enumerate(rank2)}

    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = rank1[i]
            b = rank1[j]
            if pos2[a] > pos2[b]:
                distance += 1

    return distance
