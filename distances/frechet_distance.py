import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def frechet_distance(curve1, curve2):
    n1, n2 = len(curve1), len(curve2)
    dp = np.full((n1, n2), -1.0)

	def recursive(i, j):
        if dp[i, j] >= 0:
            return dp[i, j]

        dist = euclidean_distance(curve1[i], curve2[j])

        if i == 0 and j == 0:
            dp[i, j] = dist
        elif i == 0:
            dp[i, j] = max(recursive(i, j - 1), dist)
        elif j == 0:
            dp[i, j] = max(recursive(i - 1, j), dist)
        else:
            dp[i, j] = max(
                min(
                    recursive(i - 1, j),
                    recursive(i, j - 1),
                    recursive(i - 1, j - 1),
                ),
                dist,
            )
        return dp[i, j]
    return recursive(n1 - 1, n2 - 1)
