# Autor: Martin Bublavý [xbubla02]

import cv2
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate metrics between 2 images.")
    parser.add_argument("--image1-path", type=str, help="Path to the first image")
    parser.add_argument("--image2-path", type=str, help="Path to the second image")
    
    return parser.parse_args()

def calculate_and_print_metrics(path1: str, path2: str):
    # Load images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError("One of the images could not be loaded. Please check the paths.")

    mse_value = np.mean((img1 - img2) ** 2)
    psnr_value = psnr(img1, img2, data_range=img2.max() - img2.min())
    ssim_value = ssim(img1, img2, channel_axis=-1)
    
    print(f"Mean Squared Error (MSE): {mse_value}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB")
    print(f"Structural Similarity Index (SSIM): {ssim_value}")

if __name__ == "__main__":
    args = parse_arguments()
    calculate_and_print_metrics(args.image1_path, args.image2_path)