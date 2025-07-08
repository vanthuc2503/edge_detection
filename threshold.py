#Set the threshold of the image just keep the white background and the pixel that has the value smaller than 50

import cv2
import numpy as np
import os
# Set the threshold value
THRESHOLD = 128

def threshold_image(input_path, output_path):
    # Read the image in grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    # Create a mask for pixels < threshold
    mask = img < THRESHOLD
    # Create a new image: set all to white
    result = np.full_like(img, 255)
    # Keep only pixels < threshold
    result[mask] = img[mask]

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Saved thresholded image to {output_path}")

if __name__ == "__main__":
    folder = 'result/ANIMATED2CLASSIC/fused'
    output_folder = 'result/ANIMATED2CLASSIC/threshold'
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(folder):
        if file.endswith('.png'):
            input_path = os.path.join(folder, file)
            output_path = os.path.join(output_folder, file)
            threshold_image(input_path, output_path)
            print(f"Processed {file}")
    print("Done")
