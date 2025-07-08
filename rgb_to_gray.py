import cv2
import os

#change rgb to gray
def rgb_to_gray(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray)

if __name__ == "__main__":
    input_folder = "ANIMATED/train/labels"
    output_folder = "ANIMATED/train/labels"
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith('.png'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            rgb_to_gray(input_path, output_path)
            print(f"Processed {file}")
    print("Done")