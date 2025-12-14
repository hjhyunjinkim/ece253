"""Process images with JPEG compression and restoration using SA-DCT and FBCNN."""
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from src.corruption.compression_transform import apply_compression, degrade_compression_image
from src.restoration.compression.restoration_algorithm_sa_dct import SADCT
from src.restoration.compression.restoration_deeplearning_run_fbcnn import infer_image as fbcnn_infer, save_image as fbcnn_save


def process_image(
    image_path: Path,
    output_dir: Path,
    quality_factor: int = 20,
    save_compressed: bool = True,
    save_sadct: bool = True,
    save_fbcnn: bool = True,
    fbcnn_model_path: str = None
):
    """Process a single image: compress and restore.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        quality_factor: JPEG quality factor for compression
        save_compressed: Whether to save compressed image
        save_sadct: Whether to save SA-DCT restored image
        save_fbcnn: Whether to save FBCNN restored image
        fbcnn_model_path: Optional path to FBCNN model weights
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        # Create output subdirectories
        category = image_path.parent.name
        output_category_dir = output_dir / category
        output_category_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply JPEG compression
        compressed_array = apply_compression(image_rgb, quality_factor=quality_factor, random=False)
        
        # Save compressed image
        if save_compressed:
            compressed_bgr = cv2.cvtColor(compressed_array, cv2.COLOR_RGB2BGR)
            compressed_path = output_category_dir / f"{image_path.stem}_compressed_qf{quality_factor}.jpg"
            cv2.imwrite(str(compressed_path), compressed_bgr)
        
        # Restore with SA-DCT
        if save_sadct:
            sa_dct = SADCT(block_size=8)
            sadct_restored = sa_dct.deblock(compressed_array)
            sadct_bgr = cv2.cvtColor(sadct_restored, cv2.COLOR_RGB2BGR)
            sadct_path = output_category_dir / f"{image_path.stem}_sadct_restored.jpg"
            cv2.imwrite(str(sadct_path), sadct_bgr)
        
        # Restore with FBCNN
        if save_fbcnn:
            # Save compressed image temporarily for FBCNN
            temp_compressed_path = output_category_dir / f"{image_path.stem}_temp_compressed.jpg"
            compressed_bgr = cv2.cvtColor(compressed_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(temp_compressed_path), compressed_bgr)
            
            # Run FBCNN inference
            fbcnn_restored = fbcnn_infer(str(temp_compressed_path), path_ckpt=fbcnn_model_path or "")
            
            # Save FBCNN restored image
            fbcnn_path = output_category_dir / f"{image_path.stem}_fbcnn_restored.jpg"
            fbcnn_save(fbcnn_restored, str(fbcnn_path))
            
            # Clean up temp file
            if temp_compressed_path.exists():
                temp_compressed_path.unlink()
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_directory(
    images_dir: Path,
    output_dir: Path,
    quality_factor: int = 20,
    save_compressed: bool = True,
    save_sadct: bool = True,
    save_fbcnn: bool = True,
    fbcnn_model_path: str = None,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
):
    """Process all images in a directory recursively.
    
    Args:
        images_dir: Root directory containing images
        output_dir: Directory to save outputs
        quality_factor: JPEG quality factor for compression
        save_compressed: Whether to save compressed images
        save_sadct: Whether to save SA-DCT restored images
        save_fbcnn: Whether to save FBCNN restored images
        fbcnn_model_path: Optional path to FBCNN model weights
        extensions: Image file extensions to process
    """
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.rglob(f"*{ext}"))
        image_files.extend(images_dir.rglob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    print(f"Quality factor: {quality_factor}")
    print(f"SA-DCT restoration: {'Yes' if save_sadct else 'No'}")
    print(f"FBCNN restoration: {'Yes' if save_fbcnn else 'No'}")
    if fbcnn_model_path:
        print(f"FBCNN model path: {fbcnn_model_path}")
    print("-" * 60)
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        process_image(
            image_path,
            output_dir,
            quality_factor=quality_factor,
            save_compressed=save_compressed,
            save_sadct=save_sadct,
            save_fbcnn=save_fbcnn,
            fbcnn_model_path=fbcnn_model_path
        )
    
    print(f"\nProcessing complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Process images with JPEG compression and restoration"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data",
        help="Directory containing images to process (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/compression_restored",
        help="Directory to save results (default: data/compression_restored)"
    )
    parser.add_argument(
        "--quality-factor",
        type=int,
        default=20,
        help="JPEG quality factor for compression (default: 20)"
    )
    parser.add_argument(
        "--no-compressed",
        action="store_true",
        help="Skip saving compressed images"
    )
    parser.add_argument(
        "--no-sadct",
        action="store_true",
        help="Skip SA-DCT restoration"
    )
    parser.add_argument(
        "--no-fbcnn",
        action="store_true",
        help="Skip FBCNN restoration"
    )
    parser.add_argument(
        "--fbcnn-model",
        type=str,
        default=None,
        help="Path to FBCNN model weights (optional)"
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory '{images_dir}' does not exist")
        return
    
    process_directory(
        images_dir=images_dir,
        output_dir=output_dir,
        quality_factor=args.quality_factor,
        save_compressed=not args.no_compressed,
        save_sadct=not args.no_sadct,
        save_fbcnn=not args.no_fbcnn,
        fbcnn_model_path=args.fbcnn_model
    )


if __name__ == "__main__":
    main()
