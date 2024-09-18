import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

features = np.array([
    [1, 0, 1, 0, 1], # Movie 1: Action and Science Fiction
    [0, 1, 0, 1, 0], # Movie 2: Comedy
    [1, 0, 0, 0, 1], # Movie 3: Action and Horror
    [0, 0, 0, 1, 1]  # Movie 4: Horror
])

user_profile = np.array([1, 1, 0, 0, 0])
similarities = cosine_similarity([user_profile], features)
similarities = similarities[0]
recommended_movies = np.argsort(similarities)[::-1]
print("Similarity Scores: ", similarities)
# [0.4082, 0.5, 0.5, 0.]
print("Recommended Movies: ", recommended_movies)
# Recommended:
# 1. Movie 3
# 2. Movie 2
# 3. Movie 1
# 4. Movie 4