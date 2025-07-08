from PIL import Image
import os
import glob

def split_image(image_path, output_folder):
    image = Image.open(image_path)
    width, height = image.size
    new_width = width // 2
    new_height = height // 2
    image_list = []
    for i in range(2):
        for j in range(2):
            box = (i * new_width, j * new_height, (i + 1) * new_width, (j + 1) * new_height)
            cropped_image = image.crop(box)
            cropped_image.save(f"{output_folder}/cropped_{i}_{j}.png")
            image_list.append(cropped_image)
    return image_list

def merge_image(image_list, output_folder):
    width, height = image_list[0].size
    new_width = width * 2
    new_height = height * 2
    merged_image = Image.new('RGB', (new_width, new_height))
    for i in range(2):
        for j in range(2):
            merged_image.paste(image_list[i * 2 + j], (i * width, j * height))
    merged_image.save(f"{output_folder}/merged.png")

def merge_image_list(input_folder, output_folder):
    """
    Merge multiple images from input folder into a single image.
    Assumes images are named in a pattern that indicates their position.
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        exit(1)
    
    # Sort image files to ensure proper order
    image_files.sort()
    
    # Create a new image to hold the merged result
    width, height = Image.open(image_files[0]).size
    new_width = width * 2
    new_height = height * 2
    merged_image = Image.new('RGB', (new_width, new_height))
    for i in range(2):
        for j in range(2):
            image_path = image_files[i * 2 + j]
            img = Image.open(image_path)
            merged_image.paste(img, (i * width, j * height))
    merged_image.save(f"{output_folder}/merged.png")
def rgb_to_binary(input_path, output_path):
    image = Image.open(input_path)
    image = image.convert('L')
    image.save(output_path)
    print(f"Binary image saved to {output_path}")

if __name__ == "__main__":
    # input_folder = "data"
    # output_folder = "data_png_split"
    # os.makedirs(output_folder, exist_ok=True)
    
    # # Get all image files in the input folder
    # image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    # image_files = []
    # for ext in image_extensions:
    #     image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # if not image_files:
    #     print(f"No image files found in {input_folder}")
    #     exit(1)
    
    # # Process the first image file found
    # first_image = image_files[0]
    # print(f"Processing image: {first_image}")
    # image_list = split_image(first_image, output_folder)
    # print(f"Split images saved to {output_folder}")


    # input_folder = "result/ANIMATED2CLASSIC/fused"
    # output_folder = "result/ANIMATED2CLASSIC/fused_merge"
    # os.makedirs(output_folder, exist_ok=True)
    # merge_image_list(input_folder, output_folder)

    input_path = "result/ANIMATED2CLASSIC/fused_merge/merged.png"
    output_path = "result/ANIMATED2CLASSIC/fused/1_merge_binary.png"
    rgb_to_binary(input_path, output_path)