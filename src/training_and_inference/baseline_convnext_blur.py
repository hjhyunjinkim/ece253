import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(
        description="ConvNeXt evaluation on clean/corrupted/restored (ImageNet subset)."
    )
    parser.add_argument(
        "--clean_root",
        type=str,
        required=True,
        help="Root folder with CLEAN images (ImageFolder).",
    )
    parser.add_argument(
        "--corrupt_root",
        type=str,
        required=True,
        help="Root folder with CORRUPTED images (ImageFolder).",
    )
    parser.add_argument(
        "--restored_root",
        type=str,
        required=True,
        help="Root folder with RESTORED images (ImageFolder).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--class_mapping",
        type=str,
        default=None,
        help="Optional JSON file mapping folder names to ImageNet class indices.",
    )
    return parser.parse_args()


def create_class_mapping(folder_names, mapping_file=None):
    """
    Create mapping from local folder class indices to ImageNet-1K indices.
    
    folder_names: List of folder names from ImageFolder (e.g., ['bagel', 'pizza', ...])
    mapping_file: Optional JSON file with custom mapping
    
    Returns: dict mapping local_idx -> imagenet_idx
    """
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            custom_map = json.load(f)
        # Assuming custom_map is {folder_name: imagenet_idx}
        class_to_idx = {}
        for i, folder_name in enumerate(folder_names):
            if folder_name in custom_map:
                class_to_idx[i] = custom_map[folder_name]
            else:
                raise ValueError(f"Folder {folder_name} not found in mapping file")
        return class_to_idx
    
    # Load ImageNet-1K class labels from HuggingFace model
    model_name = "facebook/convnext-tiny-224"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Get ImageNet class names: id2label is {0: 'tench, Tinca tinca', 1: 'goldfish, ...', ...}
    id2label = model.config.id2label
    
    # Build a reverse mapping: normalize labels to handle matching
    label_to_id = {}
    for idx, label_str in id2label.items():
        # ImageNet labels often have format "primary_name, secondary_name, ..."
        # Split by comma and store all variants
        variants = [v.strip().lower() for v in label_str.split(',')]
        for variant in variants:
            if variant not in label_to_id:
                label_to_id[variant] = idx
    
    # Map folder names to ImageNet indices
    class_to_idx = {}
    unmatched = []
    
    for i, folder_name in enumerate(folder_names):
        folder_lower = folder_name.lower().strip()
        
        if folder_lower in label_to_id:
            class_to_idx[i] = label_to_id[folder_lower]
        else:
            # Try to find partial matches
            found = False
            for label, idx in label_to_id.items():
                if folder_lower in label or label in folder_lower:
                    class_to_idx[i] = idx
                    found = True
                    break
            
            if not found:
                unmatched.append(folder_name)
    
    if unmatched:
        print(f"\nWARNING: Could not automatically map {len(unmatched)} classes:")
        for name in unmatched[:10]:  # Show first 10
            print(f"  - {name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")
        
        print("\nSearching ImageNet labels for similar matches...")
        for name in unmatched[:5]:
            name_lower = name.lower()
            print(f"\n'{name}' - possible matches:")
            matches = []
            for label, idx in label_to_id.items():
                if any(word in label for word in name_lower.split('_')):
                    matches.append((idx, id2label[idx]))
            for idx, label in matches[:3]:
                print(f"  [{idx}] {label}")
        
        raise ValueError(
            f"\nCould not map all folder names to ImageNet classes.\n"
            f"Please create a JSON mapping file with format:\n"
            f'{{\n  "{unmatched[0]}": <imagenet_idx>,\n  ...\n}}\n'
            f"Use --class_mapping to provide the file."
        )
    
    return class_to_idx


class HFTransform:
    """Transform using HuggingFace image processor."""
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, img):
        # HF processor expects PIL Image
        inputs = self.processor(images=img, return_tensors="pt")
        # Return the processed image tensor (remove batch dim)
        return inputs['pixel_values'].squeeze(0)


def build_loader(root, batch_size, num_workers, transform):
    dataset = datasets.ImageFolder(root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset.classes


def evaluate(model, dataloader, device, class_mapping, desc="Eval"):
    """
    Evaluate model on subset of ImageNet classes.
    
    class_mapping: dict mapping local class idx -> ImageNet-1K idx
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Get model predictions (1000 classes)
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)

            # Map local labels to ImageNet indices
            imagenet_labels = torch.tensor(
                [class_mapping[label.item()] for label in labels],
                device=device
            )

            correct += torch.sum(preds == imagenet_labels).item()
            total += inputs.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"{desc} accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load HuggingFace model and processor
    model_name = "facebook/convnext-tiny-224"
    print(f"Loading model: {model_name}")
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    
    transform = HFTransform(processor)

    # Build data loaders
    loader_clean, classes_clean = build_loader(
        args.clean_root, args.batch_size, args.num_workers, transform
    )
    loader_corrupt, classes_corrupt = build_loader(
        args.corrupt_root, args.batch_size, args.num_workers, transform
    )
    loader_restored, classes_restored = build_loader(
        args.restored_root, args.batch_size, args.num_workers, transform
    )

    # Check class consistency
    print(f"\nFound {len(classes_clean)} classes:")
    print(f"Classes: {classes_clean}")

    if not (classes_clean == classes_corrupt == classes_restored):
        raise ValueError("Class lists differ between datasets!")

    # Create mapping from local indices to ImageNet-1K indices
    print("\nCreating class mapping to ImageNet-1K indices...")
    class_mapping = create_class_mapping(classes_clean, args.class_mapping)
    
    print(f"\nClass mapping (local -> ImageNet):")
    for i, (local_idx, imagenet_idx) in enumerate(class_mapping.items()):
        label = model.config.id2label[imagenet_idx]
        print(f"  {local_idx:2d}. {classes_clean[local_idx]:20s} -> [{imagenet_idx:3d}] {label}")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    acc_clean = evaluate(model, loader_clean, device, class_mapping, desc="Clean (R)")
    acc_corrupt = evaluate(model, loader_corrupt, device, class_mapping, desc="Corrupted (R')")
    acc_restored = evaluate(model, loader_restored, device, class_mapping, desc="Restored (R'_e)")

    print("\n" + "="*60)
    denom = acc_clean - acc_corrupt
    if abs(denom) < 1e-8:
        print("Recovery ratio undefined because R - R' ≈ 0.")
    else:
        recovery = (acc_restored - acc_corrupt) / denom
        print(f"Recovery ratio: (R'_e - R') / (R - R') = {recovery:.4f}")
        print(f"  where R'_e - R' = {acc_restored - acc_corrupt:.4f}")
        print(f"        R - R'    = {denom:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()