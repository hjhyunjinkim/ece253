"""
Image Corruption Pipeline - Blur Distortions
Generates synthetic blur distortions for robustness testing.

Supports:
- Motion blur
- Defocus blur
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse


class CorruptionGenerator:
    """Generate synthetic distortions for image robustness testing"""
    
    def __init__(self, input_dir='clean', output_dir='corrupted'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_image_paths(self):
        """Get all image paths organized by class"""
        image_paths = {}
        for class_dir in sorted(self.input_dir.iterdir()):
            if class_dir.is_dir():
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                    images.extend(list(class_dir.glob(ext)))
                if images:
                    image_paths[class_dir.name] = sorted(images)
        return image_paths
    
    def apply_motion_blur(self, image, kernel_size=31):
        """Apply motion blur to simulate camera shake"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), 
                                     np.random.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        return cv2.filter2D(image, -1, kernel)
    
    def apply_defocus_blur(self, image, kernel_size=21):
        """Apply defocus (circular) blur"""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        y, x = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing='ij')
        mask = ((x - center)**2 + (y - center)**2) <= (center**2)
        kernel[mask] = 1
        kernel = kernel / kernel.sum()
        return cv2.filter2D(image, -1, kernel)
    
    
    def generate_all(self, severities=['extreme']):
        """
        Generate blur distortions at specified severity levels.
        
        Args:
            severities: List of severity levels ['low', 'medium', 'high', 'extreme']
        """
        # Blur parameters by severity
        blur_params = {
            'low': {'motion': 15, 'defocus': 11},
            'medium': {'motion': 31, 'defocus': 21},
            'high': {'motion': 51, 'defocus': 35},
            'extreme': {'motion': 101, 'defocus': 71}
        }
        
        image_paths = self.get_image_paths()
        results = {'blur_motion': [], 'blur_defocus': []}
        
        print(f"Found {len(image_paths)} classes, {sum(len(p) for p in image_paths.values())} images")
        
        for severity in severities:
            print(f"\nGenerating {severity} severity blur distortions...")
            
            # Blur distortions
            for blur_type in ['motion', 'defocus']:
                dir_name = f'blur_{blur_type}_{severity}'
                out_dir = self.output_dir / dir_name
                out_dir.mkdir(exist_ok=True)
                
                for cls, paths in tqdm(image_paths.items(), desc=f"  {blur_type}"):
                    (out_dir / cls).mkdir(exist_ok=True)
                    for img_path in paths:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        if blur_type == 'motion':
                            corrupted = self.apply_motion_blur(img, blur_params[severity]['motion'])
                        else:
                            corrupted = self.apply_defocus_blur(img, blur_params[severity]['defocus'])
                        
                        cv2.imwrite(str(out_dir / cls / img_path.name), corrupted)
                
                results[f'blur_{blur_type}'].append(out_dir)
        
        # Save metadata
        metadata = {
            'input_dir': str(self.input_dir),
            'output_dir': str(self.output_dir),
            'num_classes': len(image_paths),
            'num_images': sum(len(p) for p in image_paths.values()),
            'classes': list(image_paths.keys()),
            'severities': severities,
            'distortions': {k: [str(d) for d in v] for k, v in results.items()},
            'parameters': {
                'blur': blur_params
            }
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Blur corruption complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Generated {len(severities)} severity levels")
        print(f"  Distortion types: motion blur, defocus blur")
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate corrupted images')
    parser.add_argument('--input_dir', type=str, default='clean',
                       help='Input directory with clean images (default: clean)')
    parser.add_argument('--output_dir', type=str, default='corrupted',
                       help='Output directory for corrupted images (default: corrupted)')
    parser.add_argument('--severities', nargs='+', 
                       choices=['low', 'medium', 'high', 'extreme'],
                       default=['extreme'],
                       help='Severity levels to generate (default: extreme)')
    args = parser.parse_args()
    
    generator = CorruptionGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    generator.generate_all(severities=args.severities)
    
    print("\n✅ Done! Next step: python restoration.py")
    print("   Expected accuracy drops: -15% to -20% for extreme blur")


if __name__ == "__main__":
    main()

