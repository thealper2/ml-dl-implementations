import numpy as np
from PIL import Image

def mean_squared_error(image1, image2):
    assert image1.shape == image2.shape, "Error"
    mse = np.mean((image1 - image2) ** 2)
    return mse

def peak_signal_to_noise_ratio(image1, image2, max_pixel=255.0):
    mse = mean_squared_error(image1, image2)
    if mse == 0:
        return float("inf")

    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value

image1 = np.array(Image.open("original.jpg").convert("L"))
image2 = np.array(Image.open("compressed.jpg").convert("L"))
psnr_value = peak_signal_to_noise_ratio(image1, image2)
print(f"PSNR: {psnr_value:.2f} dB")