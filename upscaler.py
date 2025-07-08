import os
import uuid
import argparse
import requests
import json
import time
import base64
from PIL import Image
from io import BytesIO
import fal_client
import mimetypes

def on_queue_update(update):
    """
    Callback function to handle queue updates and log messages.
    """
    if isinstance(update, fal_client.InProgress):
        if update.logs is not None:  # Check if logs is None before iterating
            for log in update.logs:
                print(log["message"])


def encode_image_to_data_uri(image_path):
    # Determine the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Could not determine the MIME type of the image.")

    # Read and encode the image to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URI
    data_uri = f"data:{mime_type};base64,{encoded_string}"
    return data_uri


def save_image_from_url(image_url, output_path=None):
    """
    Save an image from a URL to a local file.

    Args:
        image_url: URL of the image to download and save.
        output_path: Path where the image should be saved. If None, generates a path.

    Returns:
        Path where the image was saved.
    """
    # Create temp directory if needed
    if output_path is None:
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, f"upscaled_{uuid.uuid4()}.png")

    # Download and save the image
    response = requests.get(image_url)
    response.raise_for_status()

    # Save the image
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def compress_image_if_needed(input_path, max_size_mb=4):
    """
    Compress an image if it exceeds the maximum file size.

    Args:
        input_path: Path to the input image
        max_size_mb: Maximum file size in MB (default 4MB for fal.ai)

    Returns:
        Path to the compressed image (if compression was needed) or original path
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    # Check current file size
    current_size = os.path.getsize(input_path)

    if current_size <= max_size_bytes:
        return input_path

    print(
        f"Image size ({current_size} bytes) exceeds limit ({max_size_bytes} bytes). Compressing..."
    )

    # Generate compressed file path
    base_name = os.path.splitext(input_path)[0]
    ext = os.path.splitext(input_path)[1]
    compressed_path = f"{base_name}_compressed{ext}"

    # Open and compress the image
    with Image.open(input_path) as img:
        # Convert to RGB if necessary (for JPEG compression)
        if img.mode in ("RGBA", "LA"):
            # For images with alpha channel, use PNG with compression
            img.save(compressed_path, "PNG", optimize=True, compress_level=9)
        else:
            # For images without alpha, try JPEG compression first
            img.convert("RGB").save(
                compressed_path.replace(ext, ".jpg"), "JPEG", quality=85, optimize=True
            )
            compressed_path = compressed_path.replace(ext, ".jpg")

    # Check if compression was successful
    new_size = os.path.getsize(compressed_path)

    if new_size <= max_size_bytes:
        print(f"Compressed image to {new_size} bytes")
        return compressed_path
    else:
        # If still too large, try more aggressive compression
        with Image.open(input_path) as img:
            # Reduce dimensions if still too large
            width, height = img.size
            new_width = int(width * 0.8)
            new_height = int(height * 0.8)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if resized_img.mode in ("RGBA", "LA"):
                resized_img.save(
                    compressed_path, "PNG", optimize=True, compress_level=9
                )
            else:
                resized_img.convert("RGB").save(
                    compressed_path.replace(".png", ".jpg"),
                    "JPEG",
                    quality=75,
                    optimize=True,
                )
                compressed_path = compressed_path.replace(".png", ".jpg")

        final_size = os.path.getsize(compressed_path)
        print(f"Resized and compressed image to {final_size} bytes")
        return compressed_path


def upscale_image(
    input_path,
    output_path=None,
    model="esrgan",
    scale=4,
    tile=400,
):
    """
    Upscale an image using fal.ai API.

    Args:
        input_path: Path to input image or URL.
        output_path: Path where output should be saved. If None, generates a path.
        model: Model to use for upscaling ('esrgan' or 'real_esrgan').
        scale: Scale factor for upscaling.
        face_enhance: Whether to enhance faces (only for some models).
        api_key: The fal.ai API key. If None, looks for FAL_KEY environment variable.

    Returns:
        Path to saved upscaled image.
    """
    # Determine if input is a URL or local file
    is_url = input_path.startswith(("http://", "https://"))

    # For local files, check and compress if needed
    if not is_url:
        input_path = compress_image_if_needed(input_path)

    # Prepare arguments
    if is_url:
        arguments = {"image_url": input_path}
    else:
        # For local files, encode to base64
        img_base64 = encode_image_to_data_uri(input_path)
        arguments = {"image_url": img_base64}

    # Add other arguments
    # arguments["scale"] = scale
    # arguments["tile"] = tile
    # arguments["model"] = model
    arguments["output_format"] = "png"

    print(f"Submitting upscaling request to fal.ai using {model}...")
    print(f"Input: {'URL' if is_url else 'Local file'}")
    # print(f"Arguments: {arguments}")

    # Submit request to fal.ai
    try:
        result = fal_client.subscribe(
            "fal-ai/esrgan",
            arguments=arguments,
            with_logs=False,
            on_queue_update=on_queue_update,
        )
        # Get the result image URL
        # if "image" in result:
        #     result_url = result["image"]
        # elif "image_url" in result:
        #     result_url = result["image_url"]
        # else:
        #     raise KeyError("No image found in the API response")

        # Save the result
        saved_path = save_image_from_url(
            image_url=result["image"]["url"],
            output_path=output_path if output_path else None,
        )
        print(f"Upscaled image saved to: {saved_path}")

        return saved_path

    except Exception as e:
        print(f"Error during upscaling: {str(e)}")
        return None


def upscale_with_dual_alpha(
    input_path,
    output_path=None,
    model="esrgan",
    scale=4,
    tile=400,
):
    """
    Upscale an image using dual upscaling approach: one for RGB and one for alpha with B,G context.
    
    This function performs two separate upscaling operations:
    1. Standard upscaling of the original image for RGB content
    2. Upscaling of a rearranged (Alpha,Blue,Green,Alpha) image for contextual alpha
    
    Then merges the RGB from the first upscale with the alpha from the second upscale.

    Args:
        input_path: Path to input image (should be RGBA).
        output_path: Path where final output should be saved. If None, generates a path.
        model: Model to use for upscaling ('esrgan' or 'real_esrgan').
        scale: Scale factor for upscaling.
        tile: Tile size for upscaling.

    Returns:
        Path to saved final image with dual upscaled alpha.
    """
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    from PIL import ImageFilter
    
    print(f"Starting dual upscaling process for: {input_path}")
    
    # Create temp directory
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. First upscale: Standard RGB upscaling
    print("Step 1: Performing main RGB upscale...")
    main_upscaled_path = upscale_image(
        input_path=input_path,
        output_path=os.path.join(temp_dir, f"temp_main_upscaled_{uuid.uuid4()}.png"),
        model=model,
        scale=scale,
        tile=tile,
    )
    
    if not main_upscaled_path or not os.path.exists(main_upscaled_path):
        print("Error: Main RGB upscale failed")
        return None
    
    print(f"Main RGB upscale completed: {main_upscaled_path}")
    
    # 2. Prepare (Alpha,Blue,Green,Alpha) image for second upscale
    print("Step 2: Preparing (A,B,G,A) image for alpha context upscale...")
    try:
        original_img = Image.open(input_path).convert("RGBA")
    except Exception as e:
        print(f"Error opening input image: {e}")
        return None
        
    _orig_r, orig_g, orig_b, orig_a = original_img.split()
    
    # Construct (Alpha-as-Red, Blue-as-Green, Green-as-Blue, Alpha-as-Alpha)
    abg_image = Image.merge("RGBA", (orig_a, orig_b, orig_g, orig_a))
    
    # Save temporary ABG image
    abg_temp_path = os.path.join(temp_dir, f"temp_abg_{uuid.uuid4()}.png")
    abg_image.save(abg_temp_path)
    print(f"Prepared (A,B,G,A) image: {abg_temp_path}")
    
    # 3. Second upscale: Alpha context upscaling
    print("Step 3: Performing alpha context upscale...")
    alpha_upscaled_path = upscale_image(
        input_path=abg_temp_path,
        output_path=os.path.join(temp_dir, f"temp_alpha_upscaled_{uuid.uuid4()}.png"),
        model=model,
        scale=scale,
        tile=tile,
    )
    
    # Clean up temporary ABG file
    try:
        os.remove(abg_temp_path)
        print(f"Cleaned up temporary file: {abg_temp_path}")
    except OSError as e:
        print(f"Warning: Could not remove temporary file {abg_temp_path}: {e}")
    
    if not alpha_upscaled_path or not os.path.exists(alpha_upscaled_path):
        print("Warning: Alpha context upscale failed, falling back to simple alpha resize")
        # Fallback: use simple resize for alpha
        main_upscaled_img = Image.open(main_upscaled_path).convert("RGBA")
        upscaled_width, upscaled_height = main_upscaled_img.size
        resized_alpha = orig_a.resize((upscaled_width, upscaled_height), Image.Resampling.LANCZOS)
    else:
        print(f"Alpha context upscale completed: {alpha_upscaled_path}")
        # Extract the first channel (which was the original alpha, now upscaled with context)
        alpha_upscaled_img = Image.open(alpha_upscaled_path).convert("RGBA")
        resized_alpha, _, _, _ = alpha_upscaled_img.split()
    
    # 4. Load main upscaled RGB
    main_upscaled_img = Image.open(main_upscaled_path).convert("RGBA")
    r_upscaled, g_upscaled, b_upscaled, _ = main_upscaled_img.split()
    
    # 5. Process the alpha channel (smoothing/sharpening)
    print("Step 4: Processing alpha channel...")
    alpha_array = np.array(resized_alpha)
    
    try:
        # Convert to float for processing
        alpha_float = alpha_array.astype(np.float32)
        
        # Apply more gentle smoothing for better edge quality
        smoothed = ndimage.gaussian_filter(alpha_float, sigma=0.8)  # Increased sigma for smoother edges
        
        # Create unsharp mask for sharpening but with reduced intensity
        blurred = ndimage.gaussian_filter(alpha_float, sigma=2.0)  # Larger blur for gentler unsharp mask
        unsharp_mask = alpha_float - blurred
        sharpening_strength = 0.4  # Reduced from 0.8 for gentler sharpening
        sharpened = smoothed + (unsharp_mask * sharpening_strength)
        
        # Apply very gentle edge enhancement
        grad_x = ndimage.sobel(smoothed, axis=1)
        grad_y = ndimage.sobel(smoothed, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_enhancement = 0.1  # Reduced from 0.3 for gentler enhancement
        final_alpha_processed = sharpened + (gradient_magnitude * edge_enhancement)
        
        # Apply final smoothing to remove any remaining artifacts
        final_alpha_processed = ndimage.gaussian_filter(final_alpha_processed, sigma=0.3)
        
        alpha_array = np.clip(final_alpha_processed, 0, 255).astype(np.uint8)
        
    except ImportError:
        print("SciPy not available, using PIL-based processing for alpha")
        alpha_pil = Image.fromarray(alpha_array, mode='L')
        # Use gentler PIL-based smoothing
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=1.0))  # Increased blur
        alpha_pil = alpha_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))  # Reduced sharpening
        alpha_array = np.array(alpha_pil)
    
    processed_alpha = Image.fromarray(alpha_array, mode='L')
    
    # 6. Merge RGB from main upscale with processed alpha
    print("Step 5: Merging RGB and alpha...")
    final_image = Image.merge("RGBA", (r_upscaled, g_upscaled, b_upscaled, processed_alpha))
    
    # Generate output path if not provided
    if output_path is None:
        dir_name = os.path.dirname(input_path)
        base_name = os.path.basename(input_path).split(".")[0]
        output_path = os.path.join(dir_name, f"{base_name}_dual_upscaled_{uuid.uuid4()}.png")
    
    # Save the final image
    final_image.save(output_path)
    print(f"Dual upscaled image saved to: {output_path}")
    
    # 7. Apply alpha mask (make transparent areas white)
    from .utils import apply_alpha_mask_to_rgb
    final_output_path = apply_alpha_mask_to_rgb(output_path, erode_alpha=True, erosion_size=1)
    print(f"Applied alpha mask with 1px gentle erosion, final output: {final_output_path}")
    
    # Clean up intermediate files
    try:
        if os.path.exists(main_upscaled_path):
            os.remove(main_upscaled_path)
        if alpha_upscaled_path and os.path.exists(alpha_upscaled_path):
            os.remove(alpha_upscaled_path)
        print("Cleaned up intermediate upscale files")
    except OSError as e:
        print(f"Warning: Could not clean up some intermediate files: {e}")
    
    return final_output_path


def main():
    """
    Main function to handle upscaling images using fal.ai API with predefined parameters.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # dotenv package not installed, will continue without it

    api_key = os.environ.get("FAL_KEY")
    if not api_key:
        raise ValueError(
            "fal.ai API key is required. Set it as an argument or as FAL_KEY environment variable in .env file."
        )
    # Define parameters directly in code
    input_path = "data/25_upscaled_new.png"  # Input image path or URL
    output_path = "data/25_upscaled_new_2.png"  # Output image path

    scale = 4  # Scale factor
    tile = 400

    # Call the upscale function with the defined parameters
    output_path = upscale_image(
        input_path=input_path,
        output_path=output_path,
        model="RealESRGAN_x4plus_anime_6B",  # "RealESRGAN_x4plus"
        scale=scale,
        tile=tile,
    )
    # save fal url to a file


if __name__ == "__main__":
    main()
