import numpy as np

def cosine_similarity(A, B):
	dot_product = np.dot(A, B)
	norm_A = np.linalg.norm(A)
	norm_B = np.linalg.norm(B)
	cosine_sim = dot_product / (norm_A * norm_B)
	return cosine_sim
