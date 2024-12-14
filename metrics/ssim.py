import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

def structural_similarity_index_measure(image1, image2, k1=0.01, k2=0.03, window_size=11, L=255):
    assert image1.shape == image2.shape, "Error"

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    window = np.ones((window_size, window_size)) / (window_size ** 2)
    mu1 = gaussian_filter(image1, sigma=1.5)
    mu2 = gaussian_filter(image2, sigma=1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(image1 ** 2, sigma=1.5) - mu1_sq
    sigam2_sq = gaussian_filter(image2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(image * image2, sigma=1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_value = np.mean(ssim_map)
    return ssim_value

image1 = np.array(Image.open("original.jpg").convert("L"))
image2 = np.array(Image.open("compressed.jpg").convert("L"))
ssim_value = structural_similarity_index_measure(image1, image2)
print(f"SSIM: {ssim_value:.2f}")