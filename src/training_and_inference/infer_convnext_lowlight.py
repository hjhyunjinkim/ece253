import io
import os
import torch
import requests

from PIL import Image
from datasets import load_dataset
from transformers import ConvNextImageProcessor, ConvNextForImageClassification

def load_model(model_name):
    """
    Downloads and loads the pre-trained ConvNeXt model and processor 
    from Hugging Face.
    """
    print("Downloading model and processor... (this may take a moment)")
    processor = ConvNextImageProcessor.from_pretrained(model_name)
    model = ConvNextForImageClassification.from_pretrained(model_name)

    return processor, model


def get_sample_image(url_or_path):
    """
    Loads an image from a local path or a URL.
    """
    if url_or_path.startswith("http"):
        print(f"Downloading image from {url_or_path}...")
        response = requests.get(url_or_path, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")


def predict_class(image, processor, model):
    """
    Runs inference on the image and returns the predicted class label and score.
    """
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_idx]
    confidence_score = torch.softmax(logits, dim=-1).max().item()

    return predicted_label, predicted_label_idx, confidence_score


def infer_image(path_image, processor, model, is_print=True):
    image = get_sample_image(path_image)
    label, label_idx, score = predict_class(image, processor, model)
    
    if is_print:
        print("-" * 30)
        print(f"Prediction: {label}")
        print(f"Prediction idx: {label_idx}")
        print(f"Confidence: {score:.4f}")
        print("-" * 30)

    return label, label_idx, score


if __name__ == "__main__":
    # dir_root = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/data"
    dir_root = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/datasets3"
    path_ckpt = "/workspace/projects/Schoolwork/ECE 253 Fundamentals of Digital Image Processing/models/imagenet_finetuned/og/checkpoint-25015"
    path_ckpt = path_ckpt if len(path_ckpt) > 0 else "facebook/convnext-tiny-224"

    processor, model = load_model(path_ckpt)

    dict_result = {"og":[], "low_light": [], "classic": [], "zerodce": []}
    for condition in ["og", "low_light", "classic", "zerodce"]:
        count_total = 0
        count_correct = 0
        condition_subdir = os.path.join(dir_root, condition, "val")
        for subdir in os.listdir(condition_subdir):
            subdir_path = os.path.join(condition_subdir, subdir)

            for file in os.listdir(subdir_path):
                if not file.endswith(".jpg"):
                    continue

                count_total += 1
                filepath = os.path.join(subdir_path, file)

                label, label_idx, score = infer_image(filepath, processor, model)

                print(f"Actual label: {subdir}")
                print("-" * 30)
                if f"{label_idx:05d}" == label:
                    count_correct += 1

        accuracy = count_correct / count_total * 100
        print(f"Total accuracy: {accuracy:.2f}% ({count_correct}/{count_total})")
        dict_result[condition].append((accuracy, count_correct, count_total))
    
    for condition in ["og", "low_light", "classic", "zerodce"]:
        accuracy, count_correct, count_total = dict_result[condition][0]
        print(f"{condition} Total accuracy: {accuracy:.2f}% ({count_correct}/{count_total})") 
        
