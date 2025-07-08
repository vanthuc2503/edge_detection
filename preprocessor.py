import re
import os
import cv2
import numpy as np
import subprocess
import vtracer

def png_to_svg(input_path, output_path):
    """
    Convert a PNG image to an SVG image using rsvg-convert.
    Args:
        input_path: The path to the input PNG image
        output_path: The path where the output SVG image will be saved
    Returns:
        str: The path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(input_path,
                                output_path,
                                colormode = 'color',        # ["color"] or "binary"
                                hierarchical = 'stacked',   # ["stacked"] or "cutout"
                                mode = 'spline',            # ["spline"] "polygon", or "none"
                                filter_speckle = 14,         # default: 4
                                color_precision = 6,        # default: 6
                                layer_difference = 16,      # default: 16
                                corner_threshold = 60,      # default: 60
                                length_threshold = 4.0,     # in [3.5, 10] default: 4.0
                                max_iterations = 10,        # default: 10
                                splice_threshold = 45,      # default: 45
                                path_precision = 3          # default: 8
                                )

def png_to_svg_post_process(input_path, output_path):
    """
    Convert a PNG image to an SVG image using rsvg-convert.
    Args:
        input_path: The path to the input PNG image
        output_path: The path where the output SVG image will be saved
    Returns:
        str: The path to the output SVG file
    """
    vtracer.convert_image_to_svg_py(input_path,
                                output_path,
                                colormode = 'binary',        # ["color"] or "binary"
                                hierarchical = 'cutout',   # ["stacked"] or "cutout"
                                mode = 'polygon',            # ["spline"] "polygon", or "none"
                                filter_speckle = 4,         # default: 4
                                color_precision = 6,        # default: 6
                                layer_difference = 16,      # default: 16
                                corner_threshold = 60,      # default: 60
                                length_threshold = 4.0,     # in [3.5, 10] default: 4.0
                                max_iterations = 10,        # default: 10
                                splice_threshold = 45,      # default: 45
                                path_precision = 8         # default: 8
                                )
    
def svg_to_png(input_path, output_path):
    """
    Convert an SVG image to a PNG image using rsvg-convert.
    Args:
        input_path: The path to the input SVG image
        output_path: The path where the output PNG image will be saved
    Returns:
        str: The path to the output PNG file
    """
    try:

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        subprocess.run(['rsvg-convert', input_path, '-o', output_path], check=True)
        print(f"Successfully converted {input_path} to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting SVG to PNG: {str(e)}")
        raise
    except FileNotFoundError:
        print("rsvg-convert not found. Please install librsvg: brew install librsvg")
        raise

def resize_svg(svg):
    """
    Add viewBox with original dimensions and resize to 1024x1024
    Args:
        svg: The svg file content
    Returns:
        str: The svg file with viewBox and 1024x1024 dimensions
    """
    # Extract original width and height from SVG
    width_match = re.search(r'width="([^"]*)"', svg)
    height_match = re.search(r'height="([^"]*)"', svg)
    
    if width_match and height_match:
        original_width = width_match.group(1)
        original_height = height_match.group(1)
        
        # Add viewBox with original dimensions (0 0 width height)
        viewbox_attr = f'viewBox="0 0 {original_width} {original_height}"'
        
        # Replace width and height with 1024x1024
        svg = re.sub(r'width="[^"]*"', 'width="1024"', svg)
        svg = re.sub(r'height="[^"]*"', 'height="1024"', svg)
        
        if 'viewBox=' not in svg:
            # Add viewBox after the <svg ... tag
            svg = re.sub(r'(<svg\b[^>]*?)>', r'\1 ' + viewbox_attr + '>', svg)

        
        return svg
    
    return svg

def calculate_stroke_width(svg):
    """
    Calculate stroke-width to ensure stroke is always 1 pixel thick when displayed at 1024x1024
    Args:
        svg: The svg file
    Returns:
        float: The stroke width
    """
    viewbox_match = re.search(r'viewBox="([^"]*)"', svg)
    if not viewbox_match:
        return 1
    # Get viewbox
    viewbox = viewbox_match.group(1)
    parts = viewbox.split()
    if len(parts) >= 4:
        vb_width = float(parts[2])
        vb_height = float(parts[3])
        
        # Desired display size
        display_width = 1024
        display_height = 1024
        
        # Calculate scale
        scale_x = display_width / vb_width
        scale_y = display_height / vb_height
        
        # Get the smaller ratio to ensure stroke is not too thick
        scale = min(scale_x, scale_y)
        
        # Stroke width = 1 pixel / scale ratio
        stroke_width = 1 / scale
        
        return round(stroke_width, 6)
    
    return 1

def transform_svg_to_line(input_path, output_path):
    """
    Transform svg to line
    Args:
        file_name: The name of the svg file
    Returns:
        None
    """

    with open(input_path, 'r', encoding='utf-8') as f:
        svg = f.read()

    # # Calculate stroke-width
    # stroke_width = calculate_stroke_width(svg)
    # print(f"Calculated stroke-width: {stroke_width}")
    stroke_width = 1

    shape_tags = ['path', 'circle', 'line', 'polygon', 'ellipse']
    for tag in shape_tags:
        # Handle path with style attribute
        svg = re.sub(rf'<{tag}\s+style="[^"]*"', f'<{tag} style="fill:#FFFFFF;stroke:#000000;stroke-width:{stroke_width};" vector-effect="non-scaling-stroke"', svg)
        
        # Handle path with fill attribute (no style)
        svg = re.sub(rf'<{tag}\s+fill="[^"]*"(?!\s+style)', f'<{tag} style="fill:#FFFFFF;stroke:#000000;stroke-width:{stroke_width};" vector-effect="non-scaling-stroke"', svg)
        
        # Handle path without style and fill (add new style)
        svg = re.sub(rf'<{tag}\s+(?!style|fill)', f'<{tag} style="fill:#FFFFFF;stroke:#000000;stroke-width:{stroke_width};" vector-effect="non-scaling-stroke" ', svg)
        
        # Handle ellipse with transform before style
        if tag == 'ellipse':
            svg = re.sub(rf'<{tag}\s+transform="([^"]*)"\s+style="[^"]*"', r'<ellipse transform="\1" style="fill:#FFFFFF;stroke:#000000;stroke-width:' + str(stroke_width) + ';" vector-effect="non-scaling-stroke"', svg)
        
        # Handle individual attributes
        svg = re.sub(r'fill="[^"]*"', 'fill="#FFFFFF"', svg)
        svg = re.sub(r'stroke="[^"]*"', 'stroke="#000000"', svg)
        svg = re.sub(r'stroke-width="[^"]*"', f'stroke-width="{stroke_width}"', svg)

    # Add viewBox and resize to 1024x1024
    svg = resize_svg(svg)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg)

def rgba_to_rgb(folder_path, output_folder_path):
    """
    Convert rgba to rgb
    Args:
        folder_path: The path to the folder containing the images
        output_folder_path: The path to the folder to save the images
    Returns:
        None
    """
    os.makedirs(output_folder_path, exist_ok=True)

    count = 0
    for file in os.listdir(folder_path):
        # Only process image files
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            continue
            
        img_path = os.path.join(folder_path, file)
        new_path = os.path.join(output_folder_path, file)

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Check if image was loaded successfully
        if img is None:
            print(f"Error: Could not load image {file}")
            continue
            
        # Check if image has alpha channel (4 channels)
        if len(img.shape) != 3 or img.shape[2] != 4:
            print(f"Warning: {file} is not a 4-channel RGBA image, skipping")
            continue

        white_bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255

        alpha = img[:, :, 3]
        mask = alpha > 0

        white_bg[mask] = img[:, :, :3][mask]

        cv2.imwrite(new_path, white_bg)
        print(f"Done {file}")
        count += 1
    print(f"Total: {count}")

def invert_image(folder_path, output_folder_path):
   """
   Invert the image
   Args:
       folder_path: The path to the folder containing the images
       output_folder_path: The path to the folder to save the images
   Returns:
       None
   """
   os.makedirs(output_folder_path, exist_ok=True)
   count = 0
   for file in os.listdir(folder_path):
       # Only process image files
       if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
           continue
           
       img_path = os.path.join(folder_path, file)
       new_path = os.path.join(output_folder_path, file)
       img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
       
       # Check if image was loaded successfully
       if img is None:
           print(f"Error: Could not load image {file}")
           continue

       if img.shape[2] == 4:
           img = img[:, :, :3]
       white_mask = np.all(img >= 250, axis=2)
       result = np.zeros_like(img)
       result[~white_mask] = [255, 255, 255]
       cv2.imwrite(new_path, result)
       print(f"Done {file}")
       count += 1
   print(f"Total: {count}")

def process_svg_files(input_folder, output_folder):
    """
    Process svg files
    Args:
        input_folder: The path to the input folder
        output_folder: The path to the output folder
    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith('.svg'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace('.svg', '_line.svg'))
            transform_svg_to_line(input_path, output_path)
            print(f"Processed {file}")
    print("Done")

def preprocess_data(input_folder, output_folder):
    """
    Preprocess data
    Args:
        input_folder: The path to the input folder
        output_folder: The path to the output folder
    Returns:
        None
    """
    temp_folder = "temp"
    temp_svg_folder = os.path.join(temp_folder, "svg")
    temp_png_folder = os.path.join(temp_folder, "png")
    os.makedirs(temp_svg_folder, exist_ok=True)
    os.makedirs(temp_png_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Step1: Convert png to svg (save to temp/svg)
    print("Step1.1: Convert png to svg")
    for file in os.listdir(input_folder):
        if file.endswith('.png'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(temp_svg_folder, file.replace('.png', '_step1_1.svg'))
            png_to_svg(input_path, output_path)
            print(f"Processed {file}")
    print("Done Step1.1: PNG to SVG")
    print("--------------------------------")

    # Step2: Resize svg to 1024x1024 (in-place in temp/svg)
    print("Step1.2: Resize svg to 1024x1024")
    for file in os.listdir(temp_svg_folder):
        if file.endswith('.svg'):
            svg_path = os.path.join(temp_svg_folder, file)
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg = f.read()
            svg = resize_svg(svg)
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(svg)
            print(f"Processed {file}")
    print("Done Step1.2: Resize SVG")
    print("--------------------------------")

    # Step3: Convert svg to png (save to temp/png)
    print("Step1.3: Convert svg to png")
    for file in os.listdir(temp_svg_folder):
        if file.endswith('.svg'):
            input_path = os.path.join(temp_svg_folder, file)
            output_path = os.path.join(temp_png_folder, file.replace('.svg', '_step1_3.png'))
            svg_to_png(input_path, output_path)
            print(f"Processed {file}")
    print("Done Step1.3: SVG to PNG")
    print("--------------------------------")

    # Step4: Change rgba to rgb (save to output_folder)
    print("Step1.4: Change rgba to rgb")
    rgba_to_rgb(temp_png_folder, output_folder)
    print("Done Step1.4: RGBA to RGB")
    print("--------------------------------")

def main():
    # input_folder = "data"
    # output_folder = "data"
    # invert_image(input_folder, output_folder)
    input_path = 

if __name__ == "__main__":
    main()


