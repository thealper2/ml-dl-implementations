import numpy as np

def frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake):
    mean_diff = np.sum((mu_real - mu_fake)**2)
    cov_sqrt = np.linalg.cholesky(np.dot(sigma_real, sigma_fake))    
    fid = mean_diff + np.trace(sigma_real + sigma_fake - 2 * cov_sqrt)
    return fid