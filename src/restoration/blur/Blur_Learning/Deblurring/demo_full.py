"""
Demo MPRNet - Auto GPU Selection
Automatically uses the GPU with most free memory

USAGE:
    python demo_auto_gpu.py --input_dir images --result_dir images_restored
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from pathlib import Path
import cv2
import argparse
import gc

parser = argparse.ArgumentParser(description='Demo MPRNet - Auto GPU Selection')
parser.add_argument('--input_dir', default='./images/', type=str)
parser.add_argument('--result_dir', default='./images_restored/', type=str)
parser.add_argument('--weights', default='model_deblurring.pth', type=str)
parser.add_argument('--max_size', default=None, type=int, help='Max image size (e.g., 512)')

args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def find_best_gpu():
    """Find GPU with most free memory"""
    if not torch.cuda.is_available():
        return None
    
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"Checking {n_gpus} available GPU(s)")
    print(f"{'='*60}")
    
    best_gpu = 0
    max_free = 0
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / (1024**3)
        
        # Set device and check memory
        torch.cuda.set_device(i)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        free = total - reserved
        
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total: {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Free: {free:.2f} GB ({free/total*100:.1f}%)")
        
        if free > max_free:
            max_free = free
            best_gpu = i
    
    print(f"\n{'='*60}")
    print(f"Selected GPU {best_gpu} with {max_free:.2f} GB free")
    print(f"{'='*60}\n")
    
    if max_free < 2.0:
        print(f"⚠ WARNING: Only {max_free:.2f} GB free!")
        print(f"  MPRNet needs ~2-3 GB. Consider:")
        print(f"  1. Use --max_size 512 to resize images")
        print(f"  2. Kill other processes using GPU")
        print(f"  3. The script will try but may fail")
        print()
    
    return best_gpu

# Find best GPU
best_gpu = find_best_gpu()

if best_gpu is None:
    print("No GPU available! Exiting.")
    exit(1)

torch.cuda.set_device(best_gpu)

# Scan files
inp_dir = args.input_dir
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

print(f"Scanning directory: {inp_dir}")
inp_path = Path(inp_dir)
files = []

for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']:
    for img_path in inp_path.rglob(ext):
        rel_path = img_path.relative_to(inp_path)
        files.append((str(img_path), str(rel_path)))

files = natsorted(files, key=lambda x: x[0])

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

print(f"Found {len(files)} images\n")

# Create directories
for _, rel_path in files:
    output_subdir = Path(out_dir) / Path(rel_path).parent
    output_subdir.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading model...")
if not Path("MPRNet.py").exists():
    print("ERROR: MPRNet.py not found!")
    print("Download: wget https://raw.githubusercontent.com/swz30/MPRNet/main/Deblurring/MPRNet.py")
    exit(1)

print("Step 1: Loading model architecture...")
load_file = run_path("MPRNet.py")
model = load_file['MPRNet']()

print("Step 2: Loading weights to CPU...")
load_checkpoint(model, args.weights)

print(f"Step 3: Moving model to GPU {best_gpu}...")
try:
    model = model.cuda(best_gpu)
    model.eval()
    print("✓ Model loaded successfully")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("\n✗ FAILED: Not enough GPU memory to load model!")
        print(f"\nGPU {best_gpu} doesn't have enough free memory (~2-3 GB needed)")
        print("\nOptions:")
        print("1. Kill other processes:")
        print("   nvidia-smi  # Find processes")
        print("   kill -9 <PID>")
        print("\n2. Try another GPU if available")
        print("\n3. Wait and try again later")
        exit(1)
    else:
        raise e

# Clear cache
torch.cuda.empty_cache()
gc.collect()

allocated = torch.cuda.memory_allocated(best_gpu) / (1024**3)
reserved = torch.cuda.memory_reserved(best_gpu) / (1024**3)
print(f"\nMemory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved\n")

img_multiple_of = 8

print("Processing images...")
print("-"*60)

for i, (file_path, rel_path) in enumerate(files, 1):
    try:
        img = Image.open(file_path).convert('RGB')
        
        # Resize if needed
        if args.max_size is not None:
            w, h = img.size
            if max(w, h) > args.max_size:
                if w > h:
                    new_w = args.max_size
                    new_h = int(h * args.max_size / w)
                else:
                    new_h = args.max_size
                    new_w = int(w * args.max_size / h)
                img = img.resize((new_w, new_h), Image.LANCZOS)
        
        input_ = TF.to_tensor(img).unsqueeze(0).cuda(best_gpu)

        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = model(input_)
        
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        f = os.path.splitext(rel_path)[0]
        output_path = os.path.join(out_dir, f+'.png')
        save_img(output_path, restored)
        
        # Clear cache periodically
        if i % 5 == 0:
            torch.cuda.empty_cache()
        
        if i % 10 == 0:
            allocated = torch.cuda.memory_allocated(best_gpu) / (1024**3)
            print(f"  [{i}/{len(files)}] GPU Memory: {allocated:.2f} GB")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n⚠ OOM on image {i}: {rel_path}")
            print(f"  Try running with --max_size 512")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            raise e

torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"✓ Processed {len(files)} images")
print(f"✓ Saved to: {out_dir}")
print(f"{'='*60}\n")