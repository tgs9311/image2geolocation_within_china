import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from transformers import CLIPProcessor, CLIPModel

# Path configuration
json_path = "/root/autodl-tmp/dataset/merged_game_locations_val.json"
image_folder = ""
output_path = "/root/autodl-tmp/TUXUN/dataset_val"
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load JSON data
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

def is_valid_coord(x):
    try:
        val = float(x)
        return not np.isnan(val)
    except (ValueError, TypeError):
        return False

filtered_data = [
    item for item in raw_data
    if is_valid_coord(item.get("latitude")) and is_valid_coord(item.get("longitude"))
]
print(f"Original records: {len(raw_data)}, valid entries after filtering: {len(filtered_data)}")

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Process images
samples = []
batch_images = []
batch_infos = []

for item in tqdm(filtered_data):
    filename = item.get("image_filename")
    full_path = os.path.join(image_folder, filename)
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found. Skipping.")
        continue

    try:
        image = Image.open(full_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {filename}: {e}")
        continue

    batch_images.append(image)
    batch_infos.append(item)

    if len(batch_images) == batch_size:
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        embeddings = features.cpu().numpy()

        for emb, info in zip(embeddings, batch_infos):
            samples.append({
                "embedding": emb.astype(np.float32),
                "image_filename": info.get("image_filename", ""),
                "latitude": float(info.get("latitude")),
                "longitude": float(info.get("longitude")),
                "country": info.get("country", ""),
                "address": info.get("address", "")
            })

        batch_images.clear()
        batch_infos.clear()

# Process remaining images
if batch_images:
    inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    embeddings = features.cpu().numpy()
    for emb, info in zip(embeddings, batch_infos):
        samples.append({
            "embedding": emb.astype(np.float32),
            "image_filename": info.get("image_filename", ""),
            "latitude": float(info.get("latitude")),
            "longitude": float(info.get("longitude")),
            "country": info.get("country", ""),
            "address": info.get("address", "")
        })

# Save as HuggingFace Dataset
dataset = Dataset.from_list(samples)
dataset.save_to_disk(output_path)

print(f"\nSuccessfully processed and saved {len(dataset)} image embeddings to: {output_path}")