# ECE 253 Final Project: Low-Light Image Enhancement

All code is located in `ece253_final/src/`.

---

## How to use low-light degradation, enhancement, and inference codes

## 1. Corrupt Clean Images with Low-Light Degradation

**File**: `src/corruption/low_light_transform.py`  
**Main Function**: `degrade_lowlight_image(path_image, path_output, intensity_factor, noise_sigma, gamma)`


### Usage Example

```python
from src.corruption.low_light_transform import degrade_lowlight_image

# Degrade a single image
path_clean_image = "/path/to/clean_image.jpg"
path_degraded_output = "/path/to/degraded_image.jpg"

degrade_lowlight_image(
    path_image=path_clean_image,
    path_output=path_degraded_output,
    intensity_factor=0.5,      # Exposure reduction (0-1, lower = darker)
    noise_sigma=25,            # Gaussian noise std dev (0-255)
    gamma=2.0                  # Gamma correction power (>1 darkens)
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path_image` | str | - | Path to input clean image file |
| `path_output` | str | - | Path to save degraded image |
| `intensity_factor` | float | 0.3 | Exposure reduction factor (0.0-1.0). Lower values = darker. |
| `noise_sigma` | float | 25 | Std deviation of Gaussian noise. Higher = grainier. |
| `gamma` | float | 2.0 | Gamma correction exponent. Higher = darker non-linearly. |


### Overview
This function applies low-light degradation to clean images by combining in the following order:
- **Gamma correction**: Non-linear darkening by taking the power of gamma for each pixel value(gamma > 1 darkens the image)
- **Exposure reduction**: Linear darkening by multiplying each pixel value by intensity_factor
- **Poisson noise**: Photon noise simulation
- **Gaussian noise**: Average White Gaussian noise of N(0, noise_sigma)



## 2. Restore Images via Robust Retinex (Algorithmic)

**File**: `src/restoration/low_light/restoration_algorithm_robust_retinex.py`  
**Main Class**: `RobustRetinex`  
**Key Method**: `.enhance(img_path)`

### Usage Example

```python
from src.restoration.low_light.restoration_algorithm_robust_retinex import RobustRetinex

# Initialize the enhancer
enhancer = RobustRetinex(
    beta=0.01,              # Sparsity weight for gradients
    omega=0.01,             # Smoothness weight for reflectance
    delta=10.0,             # Noise suppression weight
    gamma_correction=2.2    # Gamma correction for illumination
)

# Enhance a low-light image
path_degraded_image = "/path/to/degraded_image.jpg"
enhanced_image = enhancer.enhance(path_degraded_image)

# Save the result
import cv2
cv2.imwrite("/path/to/restored_image.jpg", enhanced_image)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | float | 0.01 | Sparsity weight for gradient regularization. Controls edge preservation. |
| `omega` | float | 0.01 | Smoothness weight for reflectance. Higher = smoother reflectance. |
| `delta` | float | 10.0 | Noise suppression weight. Higher = more noise suppression. |
| `gamma_correction` | float | 2.2 | Gamma correction exponent for illumination adjustment. Standard: 2.2 |
| `iterations` | int | 10 | Number of iterations per optimization loop |

### Overview
Implementation of the **Robust Retinex** model, enhances low-light images using an optimization-based approach without neural networks. It decomposes images into reflectance (R), illumination (L), and Noise (N) components: `I = R * L + N`. For comparison, the standard Multiscale Retinex algorithm decomposes images into reflectance and illumination only: `I = R * L`.

Algorithm iterates for default of 10 iterations during optimization stage.

**Paper**: "Structure-Revealing Low-Light Image Enhancement Via Robust Retinex Model"  
Link: https://ieeexplore.ieee.org/document/8304597


---

## 3. Restore Images via Zero-DCE++ (Deep Learning)

**File**: `src/restoration/low_light/restoration_deeplearning_run_zero_dce.py`  
**Main Function**: `infer_image(path_image, path_ckpt="")`

### Usage Example

```python
from src.restoration.low_light.restoration_deeplearning_run_zero_dce import infer_image, save_image

# Enhance using Zero-DCE++
path_degraded_image = "/path/to/degraded_image.jpg"
path_checkpoint = "path/to/Epoch99.pth"  # Pre-trained weights

enhanced_image = infer_image(path_degraded_image, path_checkpoint)

# Save the enhanced image
save_image(enhanced_image, "/path/to/restored_image.jpg")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path_image` | str | - | Path to input low-light image |
| `path_ckpt` | str | "" | Path to checkpoint file (.pth). If empty, uses Epoch99.pth in same directory. |

### Overview
**Zero-DCE++** (Zero-Reference Deep Curve Estimation) is a lightweight neural network that learns to adjust image curves for enhancement without reference images. It is extremely lightweight at ~10k parameters, and runs extremely quickly.

The function returns a RGB numpy array of shape (H, W, 3) in float range [0, 1], so it needs to be saved using the save_image() function, or converted manually:
```python
import cv2
enhanced_uint8 = (enhanced_image * 255).astype(np.uint8)
cv2.imwrite(path_output, cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR))
```

---

## 4. Fine-tune ConvNeXt on Low-Light Images

**File**: `finetune_convnext_lowlight.py`  
**Purpose**: Train a pre-trained ConvNeXt model on ImageNet-1K dataset combined with custom low-light/restored images for classification

### Parameters

The following parameters in the `if __name__ == "__main__":` section control training:

```python
# ===== DATASET PATHS =====
dataset_custom_og = load_dataset(
    "/workspace/projects/Schoolwork/.../datasets_nums/low_light"
)
dataset_imagenet_val = load_dataset(
    "/workspace/projects/Schoolwork/.../imagenet/kaggle/valid"
)
dataset_imagenet_train = load_dataset(
    "/workspace/projects/Schoolwork/.../imagenet/kaggle/train"
)

# ===== MODEL CONFIGURATION =====
MODEL_CHECKPOINT = "facebook/convnext-tiny-224"  # Model size/variant

# ===== TRAINING HYPERPARAMETERS =====
BATCH_SIZE = 256                    # Batch size (default: 256, paper uses 4096)
LEARNING_RATE = 1e-4                # Learning rate for optimization
NUM_EPOCHS = 5                       # Number of training epochs

# ===== OUTPUT DIRECTORY =====
OUTPUT_DIR = "/path/to/save/model"  # Where to save fine-tuned model
```

### Key Modifiable Parameters

| Parameter | Current Value | Typical Range | Effect |
|-----------|---------------|---------------|--------|
| `MODEL_CHECKPOINT` | `facebook/convnext-tiny-224` | Varies | Model architecture. Larger: better accuracy but slower. Options: `convnext-tiny`, `convnext-small`, `convnext-base` |
| `BATCH_SIZE` | 256 | 32-4096 | Larger = faster training but more VRAM. Paper uses 4096. |
| `LEARNING_RATE` | 1e-4 | 1e-5 to 1e-2 | Higher = faster learning but may diverge. Lower = more stable. |
| `NUM_EPOCHS` | 5 | 1-50 | More epochs = better training but slower. |

### Dataset Structure

The script expects:
```
datasets_nums/
├── low_light/
│   ├── train/
│   │   ├── class_1/
│   │   │   └── image1.jpg
│   │   └── class_2/
│   │       └── image2.jpg
│   └── test/
│       └── ...
```

### Overview
This script:
1. Loads a pre-trained ConvNeXt model from Hugging Face
2. Combines custom low-light dataset with ImageNet-1K training data
3. Fine-tunes the model for image classification
4. Saves the fine-tuned model

### Concatenating datasets
When combining the custom dataset with the Imagenet-1K dataset, it is required to unify the feature map used between the two datasets using .cast. If this step is not included, it will throw an error.

```python
dataset_custom = load_dataset("path/to/dataset")
dataset_imagenet = load_dataset("path/to/imagenet/")
# Alternatively, the dataset can be downloaded from Huggingface. Expect 10+ hours.
dataset_imagenet = load_dataset("imagenet-1k", download_config=DownloadConfig(resume_download=True))          

dataset_custom = dataset_custom.cast(dataset_imagenet.features)
```

---

## 5. Infer Images with Trained ConvNeXt Models

**File**: `infer_convnext_lowlight.py`  
**Main Function**: `infer_image(path_image, processor, model, is_print=True)`

### Usage Example

```python
from infer_convnext_lowlight import load_model, infer_image

# Load model and processor
model_checkpoint = "/path/to/saved/model"  # or "facebook/convnext-tiny-224"
processor, model = load_model(model_checkpoint)

# Infer on a single image
path_image = "/path/to/image.jpg"
predicted_label, label_idx, confidence = infer_image(
    path_image, 
    processor, 
    model, 
    is_print=True
)

print(f"Predicted: {predicted_label}")
print(f"Confidence: {confidence:.4f}")
```

### Parameters

#### `load_model(model_name)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | str | Hugging Face model name or local path to saved model |
| **Returns** | tuple | (processor, model) - ready for inference |

#### `infer_image(path_image, processor, model, is_print=True)`
| Parameter | Type | Description |
|-----------|------|-------------|
| `path_image` | str | Path to input image (local or URL) |
| `processor` | ConvNextImageProcessor | Image processor from Hugging Face |
| `model` | ConvNextForImageClassification | Model from Hugging Face |
| `is_print` | bool | Whether to print results to console |
| **Returns** | tuple | (predicted_label, label_idx, confidence_score) |

### Overview
This script loads a fine-tuned ConvNeXt model and runs inference on images to classify them. It can test on multiple image conditions (original, low-light, enhanced with RobustRetinex, enhanced with Zero-DCE++).


### Batch Evaluation Example

```python
# Evaluate on all conditions
dir_root = "/path/to/datasets3"
path_ckpt = "/path/to/model/checkpoint-25015"

processor, model = load_model(path_ckpt)

results = {}
for condition in ["og", "low_light", "classic", "zerodce"]:
    count_total = 0
    count_correct = 0
    
    condition_dir = os.path.join(dir_root, condition, "val")
    for category in os.listdir(condition_dir):
        for image_file in os.listdir(os.path.join(condition_dir, category)):
            image_path = os.path.join(condition_dir, category, image_file)
            label, idx, score = infer_image(image_path, processor, model, is_print=False)
            
            count_total += 1
            if prediction_matches_ground_truth:  # Your logic here
                count_correct += 1
    
    accuracy = (count_correct / count_total) * 100
    print(f"{condition}: {accuracy:.2f}% ({count_correct}/{count_total})")
```

---

## Directory Structure

```
ece253_final/
├── README.md (this file)
├── finetune_convnext_lowlight.py      # Fine-tune ConvNeXt
├── infer_convnext_lowlight.py         # Run ConvNeXt inference
├── src/
│   ├── corruption/
│   │   └── low_light_transform.py     # Degrade images
│   └── restoration/
│       └── low_light/
│           ├── Epoch99.pth                                 # Zero-DCE++ checkpoint
│           ├── restoration_algorithm_robust_retinex.py     # Retinex enhancement
│           ├── restoration_deeplearning_run_zero_dce.py    # Zero-DCE++ enhancement
│           └── zero_dce_model.py                           # Zero-DCE++ model architecture
│   └── training_and_inference/
│       ├── finetune_convnext_lowlight.py                   # Fine-tune ConvNeXt
│       └── infer_convnext_lowlight.py                      # Run ConvNeXt inference
├── data/                              # Input images
├── models/                            # Saved model checkpoints
└── datasets/                          # Training/val datasets
```

---

## Notes

