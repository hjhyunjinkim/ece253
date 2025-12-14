"""Fine-tune ConvNeXt on compression-restored images from the images folder."""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import timm
from torchvision import transforms

from src.corruption.compression_transform import apply_compression
from src.restoration.compression.restoration_algorithm_sa_dct import SADCT
from src.restoration.compression.restoration_deeplearning_run_fbcnn import infer_image as fbcnn_infer
import cv2
import numpy as np
from PIL import Image


def get_categories_from_images(images_dir: Path):
    """Get all category names from images directory."""
    categories = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
    return categories


def process_images_for_training(images_dir: Path, output_dir: Path, quality_factor: int = 20, 
                                use_fbcnn: bool = False, fbcnn_model_path: str = None):
    """Process images: compress and restore for training."""
    print("Processing images for training...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    categories = get_categories_from_images(images_dir)
    sa_dct = SADCT(block_size=8)
    
    all_images = []
    for category in categories:
        category_dir = images_dir / category
        if not category_dir.exists():
            continue
            
        output_category_dir = output_dir / category
        output_category_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in category_dir.glob('*.jpg'):
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Compress
                compressed = apply_compression(image_rgb, quality_factor=quality_factor, random=False)
                
                # Restore with FBCNN (preferred) or SA-DCT
                if use_fbcnn:
                    # Save temp compressed image for FBCNN
                    temp_path = output_category_dir / f"{img_path.stem}_temp.jpg"
                    compressed_bgr = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(temp_path), compressed_bgr)
                    
                    # Restore with FBCNN
                    restored = fbcnn_infer(str(temp_path), path_ckpt=fbcnn_model_path or "")
                    restored = (restored * 255).astype(np.uint8)
                    
                    # Clean up temp
                    temp_path.unlink()
                else:
                    # Use SA-DCT
                    restored = sa_dct.deblock(compressed)
                
                # Save restored image
                restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                output_path = output_category_dir / f"{img_path.stem}_restored.jpg"
                cv2.imwrite(str(output_path), restored_bgr)
                
                all_images.append(str(output_path))
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"Processed {len(all_images)} images")
    return all_images


def create_dataset_from_directory(images_dir: Path, categories: list, image_size=(224, 224), is_train: bool = True):
    """Create dataset from directory with category folders."""
    image_paths = []
    for category in categories:
        category_dir = images_dir / category
        if category_dir.exists():
            # Get restored images
            for img_path in category_dir.glob('*_restored.jpg'):
                image_paths.append((str(img_path), categories.index(category)))
            # Also get original images if no restored ones
            if len([p for p in category_dir.glob('*_restored.jpg')]) == 0:
                for img_path in category_dir.glob('*.jpg'):
                    if '_compressed' not in img_path.name:
                        image_paths.append((str(img_path), categories.index(category)))
    
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    # Transforms
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = ImageDataset(image_paths, transform)
    return dataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ConvNeXt on compression-restored images")
    parser.add_argument("--images-dir", type=str, default="images", help="Directory containing original images")
    parser.add_argument("--output-dir", type=str, default="data/compression_restored", help="Directory for processed images")
    parser.add_argument("--checkpoint-dir", type=str, default="src/restoration/compression", help="Directory to save checkpoint")
    parser.add_argument("--quality-factor", type=int, default=20, help="JPEG quality factor")
    parser.add_argument("--use-fbcnn", action="store_true", help="Use FBCNN for restoration (default: SA-DCT)")
    parser.add_argument("--fbcnn-model", type=str, default=None, help="Path to FBCNN model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup paths
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not images_dir.exists():
        print(f"Error: Images directory '{images_dir}' does not exist")
        return
    
    # Get categories
    categories = get_categories_from_images(images_dir)
    num_classes = len(categories)
    print(f"Found {num_classes} categories: {categories[:5]}... (showing first 5)")
    
    # Process images if output directory doesn't exist or is empty
    if not output_dir.exists() or len(list(output_dir.rglob('*.jpg'))) == 0:
        print("Processing images (compression + restoration)...")
        process_images_for_training(
            images_dir, 
            output_dir, 
            quality_factor=args.quality_factor,
            use_fbcnn=args.use_fbcnn,
            fbcnn_model_path=args.fbcnn_model
        )
    else:
        print(f"Using existing processed images in {output_dir}")
    
    # Create datasets
    print("Creating datasets...")
    full_dataset = create_dataset_from_directory(output_dir, categories, is_train=True)
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating ConvNeXt model...")
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    model.to(device)
    print(f"Model created with {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "Epoch99.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'categories': categories
            }, checkpoint_path)
            print(f"Saved best model (val_acc: {val_acc:.2f}%) to {checkpoint_path}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoint saved to: {checkpoint_dir / 'Epoch99.pth'}")


if __name__ == "__main__":
    main()
