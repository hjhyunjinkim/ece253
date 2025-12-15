"""Fine-tune ConvNeXt on compression-restored images for classification."""
import os
import torch
import evaluate
import numpy as np

from PIL import Image
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision.transforms import (
    Compose, 
    Normalize, 
    RandomResizedCrop, 
    RandomHorizontalFlip, 
    Resize, 
    CenterCrop, 
    ToTensor,
)


def preprocess_train(example_batch):
    """Apply train transforms across a batch."""
    example_batch["pixel_values"] = [
        _train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    del example_batch["image"]
    return example_batch

def preprocess_val(example_batch):
    """Apply validation transforms across a batch."""
    example_batch["pixel_values"] = [
        _val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    del example_batch["image"]
    return example_batch

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # Config
    MODEL_CHECKPOINT = "facebook/convnext-tiny-224"
    BATCH_SIZE = 256       # paper uses 4096, but may need to reduce based on VRAM
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    OUTPUT_DIR = "/path/to/models/imagenet_finetuned/compression"  # Update this path
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Dataset
    # Update these paths to your dataset locations
    # Expected structure: datasets_nums/compression/{train,test}/class_name/image.jpg
    dataset_custom_compression = load_dataset("/path/to/datasets_nums/compression")  # Update this path

    dataset_imagenet_val = load_dataset("/path/to/imagenet/kaggle/valid", split="train")  # Update this path
    dataset_imagenet_train = load_dataset("/path/to/imagenet/kaggle/train", split="train")  # Update this path

    # Cast datasets to force custom dataset to use imagenet-1k feature definition
    dataset_custom_compression = dataset_custom_compression.cast(dataset_imagenet_val.features)
    dataset_custom_train = concatenate_datasets([dataset_custom_compression["train"], dataset_imagenet_train])
    dataset_custom_test = concatenate_datasets([dataset_custom_compression["test"], dataset_imagenet_val])
    dataset_custom = {"train": dataset_custom_train, "test": dataset_custom_test}

    # Extract labels
    labels = dataset_custom["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    print(f"Classes: {labels}")

    image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    dataset_custom["train"].set_transform(preprocess_train)
    dataset_custom["test"].set_transform(preprocess_val)

    # Model
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True, 
    )
    accuracy_metric = evaluate.load("accuracy")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_custom["train"],
        eval_dataset=dataset_custom["test"],
        tokenizer=image_processor,              # save processor config with model
        compute_metrics=compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

