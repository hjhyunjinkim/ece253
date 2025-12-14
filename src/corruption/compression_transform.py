"""JPEG compression corruption transform."""
import os
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional


def apply_compression(image: np.ndarray, quality_factor: Optional[int] = None, random: bool = True) -> np.ndarray:
    """
    Apply JPEG compression corruption to an image.

    Args:
        image (numpy.ndarray): The input RGB or BGR image in uint8 format.
        quality_factor (int, optional): JPEG quality factor (1-100, lower = more compression).
                                        If None and random=True, uses random quality (5-20).
                                        If None and random=False, uses quality 10.
        random (bool): Whether to use random quality if not specified.

    Returns:
        numpy.ndarray: The compressed image in uint8 format.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # Random quality factor if not specified
    if quality_factor is None:
        quality_factor = np.random.randint(5, 21) if random else 10
    
    # Ensure quality is in valid range
    quality_factor = max(1, min(100, quality_factor))
    
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image, mode='L')
    else:
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if image.shape[2] == 3:
            # Assume it's RGB, but if it's BGR from cv2, convert
            pil_image = Image.fromarray(image, mode='RGB')
        else:
            pil_image = Image.fromarray(image, mode='RGB')
    
    # Apply JPEG compression
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality_factor)
    buffer.seek(0)
    
    # Load compressed image
    compressed_pil = Image.open(buffer)
    compressed_array = np.array(compressed_pil)
    
    return compressed_array


def degrade_compression_image(path_image: str, path_output: str, quality_factor: int = 20):
    """
    Apply JPEG compression degradation to input image, and save it to desired output path.

    Args:
        path_image (str): path to input image file
        path_output (str): path to save compressed image to
        quality_factor (int): JPEG quality factor (1-100, lower = more compression, default: 20)
    """
    assert os.path.isfile(path_image), "Input is not a file"

    image = cv2.imread(path_image)
    if image is None:
        raise ValueError(f"Could not read image from {path_image}")
    
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply compression
    compressed_image = apply_compression(image_rgb, quality_factor=quality_factor, random=False)
    
    # Convert back to BGR for OpenCV
    compressed_bgr = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path_output, compressed_bgr)
    print(f"{os.path.split(path_output)[1]} Saved!")

    return compressed_bgr


if __name__ == "__main__":
    # Example usage
    path_img = "/path/to/input/image.jpg"
    path_img_out = "/path/to/output/compressed_image.jpg"
    
    degrade_compression_image(path_img, path_img_out, quality_factor=20)

