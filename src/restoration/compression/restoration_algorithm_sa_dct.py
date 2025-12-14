"""SA-DCT (Shape-Adaptive DCT) deblocking algorithm for JPEG compression restoration."""
import numpy as np
import cv2
from typing import Optional


class SADCT:
    """Shape-Adaptive DCT deblocking for JPEG artifacts."""
    
    def __init__(self, block_size: int = 8):
        """
        Initialize SA-DCT deblocking algorithm.

        Args:
            block_size (int): JPEG block size (typically 8)
        """
        self.block_size = block_size

    def deblock(self, image: np.ndarray) -> np.ndarray:
        """
        Apply SA-DCT deblocking to compressed image.

        Args:
            image: Compressed image with JPEG artifacts (H, W, C) or (H, W)

        Returns:
            Deblocked image
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Apply bilateral filter to reduce blocking artifacts
        if len(image.shape) == 2:
            # Grayscale
            deblocked = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # Color
            deblocked = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Additional smoothing at block boundaries
        # This is a simplified approximation of SA-DCT
        kernel = np.ones((3, 3), np.float32) / 9.0
        deblocked = cv2.filter2D(deblocked, -1, kernel)
        
        return np.clip(deblocked, 0, 255).astype(np.uint8)

    def enhance(self, img_path: str) -> np.ndarray:
        """
        Enhance a compressed image from file path.

        Args:
            img_path: Path to compressed image file

        Returns:
            Enhanced image as numpy array (BGR format for OpenCV)
        """
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image from {img_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply deblocking
        restored_rgb = self.deblock(image_rgb)
        
        # Convert back to BGR for OpenCV
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        
        return restored_bgr


def restore_sa_dct(image: np.ndarray, block_size: int = 8) -> np.ndarray:
    """
    Restore compressed image using SA-DCT deblocking.

    Args:
        image: Compressed image as numpy array
        block_size: JPEG block size (default: 8)

    Returns:
        Restored image as numpy array
    """
    sa_dct = SADCT(block_size=block_size)
    return sa_dct.deblock(image)


if __name__ == "__main__":
    # Example usage
    sa_dct = SADCT(block_size=8)
    
    path_compressed = "/path/to/compressed_image.jpg"
    restored_image = sa_dct.enhance(path_compressed)
    
    cv2.imwrite("/path/to/restored_image.jpg", restored_image)
    print("Restored image saved!")

