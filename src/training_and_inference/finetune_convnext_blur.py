# import os
# import time
# import copy
# import argparse
# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets
# from transformers import ConvNextForImageClassification, ConvNextImageProcessor
# from PIL import Image


# def get_args():
#     parser = argparse.ArgumentParser(
#         description="Fine-tune ConvNeXt on blur-restored dataset (auto train/val split)."
#     )
#     parser.add_argument(
#         "--data_root",
#         type=str,
#         required=True,
#         help="Root folder containing class subfolders of BLUR-RESTORED images.",
#     )
#     parser.add_argument(
#         "--val_ratio",
#         type=float,
#         default=0.2,
#         help="Fraction of data to use for validation (0-1).",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=32,
#         help="Batch size for training and validation.",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=20,
#         help="Number of training epochs.",
#     )
#     parser.add_argument(
#         "--lr",
#         type=float,
#         default=5e-5,
#         help="Learning rate.",
#     )
#     parser.add_argument(
#         "--weight_decay",
#         type=float,
#         default=1e-4,
#         help="Weight decay for optimizer.",
#     )
#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=4,
#         help="Number of data loader workers.",
#     )
#     parser.add_argument(
#         "--model_out",
#         type=str,
#         default="convnext_blur_finetuned.pth",
#         help="Where to save the best model weights.",
#     )
#     parser.add_argument(
#         "--freeze_backbone",
#         action="store_true",
#         help="If set, freeze all layers except classifier head.",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for train/val split.",
#     )
#     # Optional extra evaluation datasets:
#     parser.add_argument(
#         "--eval_clean_root",
#         type=str,
#         default=None,
#         help="Optional: root folder of CLEAN images (for eval only).",
#     )
#     parser.add_argument(
#         "--eval_corrupt_root",
#         type=str,
#         default=None,
#         help="Optional: root folder of CORRUPTED images (for eval only).",
#     )
#     parser.add_argument(
#         "--eval_restored_root",
#         type=str,
#         default=None,
#         help="Optional: root folder of RESTORED images (for eval only).",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="facebook/convnext-tiny-224",
#         help="Hugging Face model name/path.",
#     )
#     return parser.parse_args()


# def set_seed(seed: int):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# class ImageFolderWithProcessor(datasets.ImageFolder):
#     """Custom ImageFolder that uses HuggingFace processor instead of torchvision transforms."""
    
#     def __init__(self, root, processor, augment=False):
#         super().__init__(root)
#         self.processor = processor
#         self.augment = augment
        
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         image = Image.open(path).convert('RGB')
        
#         # Apply processor
#         if self.augment:
#             # For training: add data augmentation
#             import torchvision.transforms as T
#             aug_transform = T.Compose([
#                 T.RandomHorizontalFlip(),
#                 T.RandomRotation(10),
#                 T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
#             ])
#             image = aug_transform(image)
        
#         # Process image
#         inputs = self.processor(images=image, return_tensors="pt")
#         pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
        
#         return pixel_values, target


# def build_autosplit_dataloaders(data_root, val_ratio, batch_size, num_workers, seed, processor):
#     # Base dataset to get class info
#     base_dataset = datasets.ImageFolder(data_root)
#     n = len(base_dataset)
#     indices = list(range(n))

#     set_seed(seed)
#     random.shuffle(indices)

#     split = int(n * (1.0 - val_ratio))
#     train_indices = indices[:split]
#     val_indices = indices[split:]

#     # Create datasets with processor
#     train_dataset_full = ImageFolderWithProcessor(data_root, processor, augment=True)
#     val_dataset_full = ImageFolderWithProcessor(data_root, processor, augment=False)

#     train_dataset = Subset(train_dataset_full, train_indices)
#     val_dataset = Subset(val_dataset_full, val_indices)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     class_names = base_dataset.classes
#     return train_loader, val_loader, class_names


# def build_eval_loader(root, batch_size, num_workers, processor):
#     if root is None:
#         return None

#     dataset = ImageFolderWithProcessor(root, processor, augment=False)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     return loader, dataset.classes


# def build_model(model_name, num_classes, freeze_backbone=False):
#     # Load pretrained ConvNeXt from Hugging Face
#     model = ConvNextForImageClassification.from_pretrained(
#         model_name,
#         num_labels=num_classes,
#         ignore_mismatched_sizes=True  # Allow classifier head to be replaced
#     )

#     if freeze_backbone:
#         # Freeze all parameters except the classifier
#         for name, param in model.named_parameters():
#             if not name.startswith("classifier"):
#                 param.requires_grad = False

#     return model


# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     running_corrects = 0
#     n_samples = 0

#     for inputs, labels in dataloader:
#         inputs = inputs.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)

#         optimizer.zero_grad()
#         outputs = model(inputs, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits
#         _, preds = torch.max(logits, 1)

#         loss.backward()
#         optimizer.step()

#         batch_size = inputs.size(0)
#         running_loss += loss.item() * batch_size
#         running_corrects += torch.sum(preds == labels).item()
#         n_samples += batch_size

#     epoch_loss = running_loss / n_samples
#     epoch_acc = running_corrects / n_samples
#     return epoch_loss, epoch_acc


# def eval_model(model, dataloader, criterion=None, device=None, desc="Eval"):
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0
#     n_samples = 0

#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs = inputs.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             outputs = model(inputs, labels=labels if criterion is not None else None)
#             logits = outputs.logits
            
#             if criterion is not None and outputs.loss is not None:
#                 running_loss += outputs.loss.item() * inputs.size(0)

#             _, preds = torch.max(logits, 1)
#             running_corrects += torch.sum(preds == labels).item()
#             n_samples += inputs.size(0)

#     acc = running_corrects / n_samples if n_samples > 0 else 0.0
#     if criterion is not None:
#         avg_loss = running_loss / n_samples
#         print(f"{desc} - loss: {avg_loss:.4f}  acc: {acc:.4f}")
#     else:
#         print(f"{desc} - acc: {acc:.4f}")

#     return acc


# def main():
#     args = get_args()
#     set_seed(args.seed)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Initialize processor
#     processor = ConvNextImageProcessor.from_pretrained(args.model_name)

#     # Build train/val from a single blur-restored dataset
#     train_loader, val_loader, class_names = build_autosplit_dataloaders(
#         data_root=args.data_root,
#         val_ratio=args.val_ratio,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         seed=args.seed,
#         processor=processor,
#     )
#     num_classes = len(class_names)
#     print(f"Number of classes: {num_classes}")
#     print(f"Classes: {class_names}")

#     model = build_model(args.model_name, num_classes, freeze_backbone=args.freeze_backbone).to(device)
#     criterion = nn.CrossEntropyLoss()

#     params_to_update = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.AdamW(
#         params_to_update,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_val_acc = 0.0

#     for epoch in range(args.epochs):
#         start_time = time.time()
#         print(f"\nEpoch {epoch + 1}/{args.epochs}")
#         print("-" * 40)

#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, criterion, optimizer, device
#         )
#         print(f"Train - loss: {train_loss:.4f}  acc: {train_acc:.4f}")

#         val_acc = eval_model(
#             model, val_loader, criterion=criterion, device=device, desc="Validation"
#         )

#         elapsed = time.time() - start_time
#         print(f"Epoch time: {elapsed:.1f}s")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_model_wts = copy.deepcopy(model.state_dict())
#             torch.save(best_model_wts, args.model_out)
#             print(
#                 f"*** New best model saved to {args.model_out} "
#                 f"(val acc = {val_acc:.4f})"
#             )

#     print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
#     model.load_state_dict(best_model_wts)

#     # Extra evaluation on clean / corrupted / restored sets (if provided)
#     if any([args.eval_clean_root, args.eval_corrupt_root, args.eval_restored_root]):
#         print("\n=== Extra evaluation on provided datasets ===")
#         # Blur-restored (training data root) as "restored" if eval_restored_root not given
#         restored_root = args.eval_restored_root or args.data_root

#         loaders = {}
#         names = []

#         if args.eval_clean_root:
#             loader_clean, classes_clean = build_eval_loader(
#                 args.eval_clean_root, args.batch_size, args.num_workers, processor
#             )
#             loaders["clean"] = loader_clean
#             names.append(("clean", classes_clean))

#         if args.eval_corrupt_root:
#             loader_corr, classes_corr = build_eval_loader(
#                 args.eval_corrupt_root, args.batch_size, args.num_workers, processor
#             )
#             loaders["corrupt"] = loader_corr
#             names.append(("corrupt", classes_corr))

#         loader_rest, classes_rest = build_eval_loader(
#             restored_root, args.batch_size, args.num_workers, processor
#         )
#         loaders["restored"] = loader_rest
#         names.append(("restored", classes_rest))

#         # Simple check: class lists should match
#         reference_classes = None
#         for label, cls_list in names:
#             print(f"{label} classes: {cls_list}")
#             if reference_classes is None:
#                 reference_classes = cls_list
#             elif reference_classes != cls_list:
#                 print(f"WARNING: class mismatch between datasets for {label}.")

#         acc_clean = acc_corrupt = acc_restored = None

#         if "clean" in loaders and loaders["clean"] is not None:
#             acc_clean = eval_model(
#                 model, loaders["clean"], criterion=None, device=device, desc="Clean"
#             )
#         if "corrupt" in loaders and loaders["corrupt"] is not None:
#             acc_corrupt = eval_model(
#                 model, loaders["corrupt"], criterion=None, device=device, desc="Corrupted"
#             )
#         if "restored" in loaders and loaders["restored"] is not None:
#             acc_restored = eval_model(
#                 model, loaders["restored"], criterion=None, device=device, desc="Restored"
#             )

#         if acc_clean is not None and acc_corrupt is not None and acc_restored is not None:
#             denom = acc_clean - acc_corrupt
#             if abs(denom) < 1e-8:
#                 print(
#                     "Recovery ratio undefined because clean_acc - corrupt_acc == 0."
#                 )
#             else:
#                 recovery = (acc_restored - acc_corrupt) / denom
#                 print(
#                     f"\nRecovery ratio (fine-tuned): "
#                     f"((R'_e - R') / (R - R')) = {recovery:.4f}"
#                 )


# if __name__ == "__main__":
#     main()
import os
import time
import copy
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from transformers import ConvNextForImageClassification, ConvNextImageProcessor
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ConvNeXt on blur-restored dataset (auto train/val split)."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing class subfolders of BLUR-RESTORED images.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (0-1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="convnext_blur_finetuned.pth",
        help="Where to save the best model weights.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="If set, freeze all layers except classifier head.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--class_mapping",
        type=str,
        required=True,
        help="JSON file mapping folder names to ImageNet class indices.",
    )
    # Optional extra evaluation datasets:
    parser.add_argument(
        "--eval_clean_root",
        type=str,
        default=None,
        help="Optional: root folder of CLEAN images (for eval only).",
    )
    parser.add_argument(
        "--eval_corrupt_root",
        type=str,
        default=None,
        help="Optional: root folder of CORRUPTED images (for eval only).",
    )
    parser.add_argument(
        "--eval_restored_root",
        type=str,
        default=None,
        help="Optional: root folder of RESTORED images (for eval only).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/convnext-tiny-224",
        help="Hugging Face model name/path.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_mapping(mapping_file):
    """Load class name to ImageNet index mapping from JSON file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def create_index_mapping(folder_names, name_to_imagenet_idx):
    """
    Create mapping from local folder class indices to ImageNet-1K indices.
    
    Args:
        folder_names: List of folder names from ImageFolder
        name_to_imagenet_idx: Dict mapping folder name to ImageNet index
    
    Returns: 
        dict mapping local_idx -> imagenet_idx
    """
    class_to_idx = {}
    for i, folder_name in enumerate(folder_names):
        if folder_name in name_to_imagenet_idx:
            class_to_idx[i] = name_to_imagenet_idx[folder_name]
        else:
            raise ValueError(f"Folder '{folder_name}' not found in mapping file")
    return class_to_idx


class ImageFolderWithProcessor(datasets.ImageFolder):
    """Custom ImageFolder that uses HuggingFace processor and maps labels to ImageNet indices."""
    
    def __init__(self, root, processor, class_mapping, augment=False):
        super().__init__(root)
        self.processor = processor
        self.augment = augment
        self.class_mapping = class_mapping  # local_idx -> imagenet_idx
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        
        # Apply processor
        if self.augment:
            # For training: add data augmentation
            import torchvision.transforms as T
            aug_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            ])
            image = aug_transform(image)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
        
        # Map local class index to ImageNet index
        imagenet_target = self.class_mapping[target]
        
        return pixel_values, imagenet_target


def build_autosplit_dataloaders(data_root, val_ratio, batch_size, num_workers, seed, processor, class_mapping):
    # Base dataset to get class info
    base_dataset = datasets.ImageFolder(data_root)
    n = len(base_dataset)
    indices = list(range(n))

    set_seed(seed)
    random.shuffle(indices)

    split = int(n * (1.0 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Create datasets with processor and class mapping
    train_dataset_full = ImageFolderWithProcessor(data_root, processor, class_mapping, augment=True)
    val_dataset_full = ImageFolderWithProcessor(data_root, processor, class_mapping, augment=False)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = base_dataset.classes
    return train_loader, val_loader, class_names


def build_eval_loader(root, batch_size, num_workers, processor, class_mapping):
    if root is None:
        return None, None

    dataset = ImageFolderWithProcessor(root, processor, class_mapping, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset.classes


def build_model(model_name, freeze_backbone=False):
    # Load pretrained ConvNeXt from Hugging Face
    # Keep original 1000-class head - we're mapping our labels to ImageNet indices
    model = ConvNextForImageClassification.from_pretrained(model_name)

    if freeze_backbone:
        # Freeze all parameters except the classifier
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, 1)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    return epoch_loss, epoch_acc


def eval_model(model, dataloader, criterion=None, device=None, desc="Eval"):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs, labels=labels if criterion is not None else None)
            logits = outputs.logits
            
            if criterion is not None and outputs.loss is not None:
                running_loss += outputs.loss.item() * inputs.size(0)

            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == labels).item()
            n_samples += inputs.size(0)

    acc = running_corrects / n_samples if n_samples > 0 else 0.0
    if criterion is not None:
        avg_loss = running_loss / n_samples
        print(f"{desc} - loss: {avg_loss:.4f}  acc: {acc:.4f}")
    else:
        print(f"{desc} - acc: {acc:.4f}")

    return acc


def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    print(f"Loading class mapping from: {args.class_mapping}")
    name_to_imagenet_idx = load_class_mapping(args.class_mapping)
    print(f"Loaded mapping for {len(name_to_imagenet_idx)} classes")

    # Initialize processor
    processor = ConvNextImageProcessor.from_pretrained(args.model_name)

    # Build train/val from a single blur-restored dataset
    # First, get folder names to create index mapping
    temp_dataset = datasets.ImageFolder(args.data_root)
    folder_names = temp_dataset.classes
    class_mapping = create_index_mapping(folder_names, name_to_imagenet_idx)
    
    print(f"\nClass mapping (local -> ImageNet):")
    for local_idx, imagenet_idx in class_mapping.items():
        print(f"  {local_idx:2d}. '{folder_names[local_idx]:20s}' -> ImageNet index {imagenet_idx}")

    train_loader, val_loader, class_names = build_autosplit_dataloaders(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        processor=processor,
        class_mapping=class_mapping,
    )
    num_classes = len(class_names)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = build_model(args.model_name, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params_to_update,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    for epoch in range(args.epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train - loss: {train_loss:.4f}  acc: {train_acc:.4f}")

        val_acc = eval_model(
            model, val_loader, criterion=criterion, device=device, desc="Validation"
        )

        elapsed = time.time() - start_time
        print(f"Epoch time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, args.model_out)
            print(
                f"*** New best model saved to {args.model_out} "
                f"(val acc = {val_acc:.4f})"
            )

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)

    # Extra evaluation on clean / corrupted / restored sets (if provided)
    if any([args.eval_clean_root, args.eval_corrupt_root, args.eval_restored_root]):
        print("\n" + "="*70)
        print("EXTRA EVALUATION ON PROVIDED DATASETS")
        print("="*70)
        
        # Need to create class mappings for eval datasets
        loaders = {}
        
        if args.eval_clean_root:
            temp_clean = datasets.ImageFolder(args.eval_clean_root)
            clean_mapping = create_index_mapping(temp_clean.classes, name_to_imagenet_idx)
            loader_clean, classes_clean = build_eval_loader(
                args.eval_clean_root, args.batch_size, args.num_workers, processor, clean_mapping
            )
            loaders["clean"] = loader_clean
            print(f"Clean classes: {classes_clean}")

        if args.eval_corrupt_root:
            temp_corrupt = datasets.ImageFolder(args.eval_corrupt_root)
            corrupt_mapping = create_index_mapping(temp_corrupt.classes, name_to_imagenet_idx)
            loader_corr, classes_corr = build_eval_loader(
                args.eval_corrupt_root, args.batch_size, args.num_workers, processor, corrupt_mapping
            )
            loaders["corrupt"] = loader_corr
            print(f"Corrupt classes: {classes_corr}")

        if args.eval_restored_root:
            temp_restored = datasets.ImageFolder(args.eval_restored_root)
            restored_mapping = create_index_mapping(temp_restored.classes, name_to_imagenet_idx)
            loader_rest, classes_rest = build_eval_loader(
                args.eval_restored_root, args.batch_size, args.num_workers, processor, restored_mapping
            )
            loaders["restored"] = loader_rest
            print(f"Restored classes: {classes_rest}")

        acc_clean = acc_corrupt = acc_restored = None

        if "clean" in loaders and loaders["clean"] is not None:
            acc_clean = eval_model(
                model, loaders["clean"], criterion=None, device=device, desc="Clean (R)      "
            )
        if "corrupt" in loaders and loaders["corrupt"] is not None:
            acc_corrupt = eval_model(
                model, loaders["corrupt"], criterion=None, device=device, desc="Corrupted (R') "
            )
        if "restored" in loaders and loaders["restored"] is not None:
            acc_restored = eval_model(
                model, loaders["restored"], criterion=None, device=device, desc="Restored (R'_e)"
            )

        if acc_clean is not None and acc_corrupt is not None and acc_restored is not None:
            print("\n" + "="*70)
            denom = acc_clean - acc_corrupt
            if abs(denom) < 1e-8:
                print("Recovery ratio undefined because R - R' ≈ 0.")
            else:
                recovery = (acc_restored - acc_corrupt) / denom
                print(f"Recovery ratio (fine-tuned): (R'_e - R') / (R - R') = {recovery:.4f}")
                print(f"  Numerator   (R'_e - R') = {acc_restored - acc_corrupt:.4f}")
                print(f"  Denominator (R - R')    = {denom:.4f}")
            print("="*70)


if __name__ == "__main__":
    main()