import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='./images/', type=str, help='Input images with class subfolders')
parser.add_argument('--result_dir', default='./samples/output/', type=str, help='Directory for results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])
parser.add_argument('--downsample', default=1, type=float, help='Downsample factor (e.g., 2 = half size, 0.5 = double size)')
parser.add_argument('--max_size', default=None, type=int, help='Maximum dimension (will downsample if larger)')

args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def calculate_downsample_size(original_size, downsample_factor=1, max_size=None):
    """Calculate the size to downsample to"""
    w, h = original_size
    
    if max_size is not None:
        # Calculate downsample factor based on max_size
        max_dim = max(w, h)
        if max_dim > max_size:
            downsample_factor = max_dim / max_size
    
    if downsample_factor != 1:
        new_w = int(w / downsample_factor)
        new_h = int(h / downsample_factor)
        # Make sure dimensions are at least 8 pixels and even numbers
        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)
        return (new_w, new_h), downsample_factor
    
    return original_size, 1

task = args.task
inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

# Get all class folders
class_folders = [d for d in os.listdir(inp_dir) 
                 if os.path.isdir(os.path.join(inp_dir, d))]

if len(class_folders) == 0:
    raise Exception(f"No class folders found at {inp_dir}")

# Collect all files with their relative paths
all_files = []
for class_folder in class_folders:
    class_path = os.path.join(inp_dir, class_folder)
    files = natsorted(glob(os.path.join(class_path, '*.jpg'))
                    + glob(os.path.join(class_path, '*.JPG'))
                    + glob(os.path.join(class_path, '*.png'))
                    + glob(os.path.join(class_path, '*.PNG')))
    
    # Store tuples of (full_path, class_folder)
    for file in files:
        all_files.append((file, class_folder))
    
    # Create corresponding output folder
    os.makedirs(os.path.join(out_dir, class_folder), exist_ok=True)

if len(all_files) == 0:
    raise Exception(f"No image files found in class folders at {inp_dir}")

print(f"Found {len(all_files)} images across {len(class_folders)} classes")
if args.downsample != 1:
    print(f"Downsample factor: {args.downsample}")
if args.max_size is not None:
    print(f"Max size: {args.max_size}")

# Load corresponding model architecture and weights
load_file = run_path(os.path.join(task, "MPRNet.py"))
model = load_file['MPRNet']()
model.cuda()

weights = os.path.join(task, "pretrained_models", "model_" + task.lower() + ".pth")
load_checkpoint(model, weights)
model.eval()

img_multiple_of = 8

for idx, (file_, class_folder) in enumerate(all_files):
    print(f"Processing {idx+1}/{len(all_files)}: {class_folder}/{os.path.basename(file_)}")
    
    # Load image
    img = Image.open(file_).convert('RGB')
    original_size = img.size  # (width, height)
    print(f"  Original size: {original_size}")
    
    # Calculate downsample size
    process_size, actual_factor = calculate_downsample_size(
        original_size, 
        args.downsample, 
        args.max_size
    )
    
    # Downsample if needed
    if process_size != original_size:
        img_downsampled = img.resize(process_size, Image.LANCZOS)
        print(f"  Downsampled to: {process_size} (factor: {actual_factor:.2f})")
    else:
        img_downsampled = img
        print(f"  No downsampling needed")
    
    # Convert to tensor
    input_ = TF.to_tensor(img_downsampled).unsqueeze(0).cuda()

    # Pad the input if not_multiple_of 8
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # Process with model
    with torch.no_grad():
        restored = model(input_)
    restored = restored[0]
    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:, :, :h, :w]

    # Convert to numpy
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])
    
    # Restore to original size if we downsampled
    if process_size != original_size:
        restored = cv2.resize(restored, original_size, interpolation=cv2.INTER_LANCZOS4)
        print(f"  Restored to original size: {original_size}")

    # Save to corresponding class folder in output directory
    f = os.path.splitext(os.path.basename(file_))[0]
    output_path = os.path.join(out_dir, class_folder, f + '.png')
    save_img(output_path, restored)
    
    # Clear GPU memory
    del input_, restored
    torch.cuda.empty_cache()

print(f"\nProcessing complete! Files saved at {out_dir}")
print(f"Processed {len(class_folders)} classes: {', '.join(class_folders)}")