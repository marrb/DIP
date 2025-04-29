import argparse
import os
import cv2
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument("--image-path", type=str, help="Path to the image file")
    parser.add_argument("--output-path", type=str, help="Path to save the output.")
    parser.add_argument("--gaussian-blur", action='store_true', help="Apply Gaussian blur to the image")
    parser.add_argument("--bicubic", action='store_true', help="Apply bicubic interpolation to the image")
    parser.add_argument("--move", action='store_true', help="Moveimage by 1 pixel on x and y axis")
    
    return parser.parse_args()

def apply_gaussian_blur(image_path: str, output_path: str):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(output_path, blurred_image)
    
def apply_bicubic_interpolation(image_path: str, output_path: str):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Upscale
    upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # Downscale back
    downscaled = cv2.resize(upscaled, (width, height), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(output_path, downscaled)
    
def move_image(image_path: str, output_path: str):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Define the affine matrix for translation
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    
    # Apply affine transformation
    moved_image = cv2.warpAffine(image, M, (width, height), borderValue=(0, 0, 0))
    
    cv2.imwrite(output_path, moved_image)


if __name__ == "__main__":
    args = parse_arguments()
    image_path = args.image_path
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    if args.gaussian_blur:
        output_path = os.path.join(args.output_path, "blurred_image.jpg")
        apply_gaussian_blur(image_path, output_path)
        exit(0)
        
    if args.bicubic:
        output_path = os.path.join(args.output_path, "bicubic_image.jpg")
        apply_bicubic_interpolation(image_path, output_path)
        exit(0)
        
    if args.move:
        output_path = os.path.join(args.output_path, "moved_image.jpg")
        move_image(image_path, output_path)
        exit(0)
        