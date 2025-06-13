import torch
import json
import re
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
from src.model import CountryModel, CityModel, ProvinceModel
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm
import faiss
from datasets import Dataset
import io
from paddleocr import PaddleOCR
import openai
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import os
import random
import base64
import mimetypes
from dotenv import load_dotenv

def get_image_media_type(image_path):
    """Determines the media type of an image file."""
    mimetype, _ = mimetypes.guess_type(image_path)
    if mimetype and mimetype.startswith("image/"):
        return mimetype
    # Fallback for common types if mimetypes doesn't guess correctly
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    return None
def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None
def generate_response(openai_client: openai.OpenAI, image_path: str, prompt_text:str) -> tuple[str | None, Dict[str, Any] | None]:
    if not openai_client:
        print("OpenAI client is not initialized. Skipping cloud call.")
        return "Skipped cloud call due to missing OpenAI client.", None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ]
        }
    ]
    # Handle image or text file input
    image_media_type = get_image_media_type(image_path)
    if image_media_type:
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{base64_image}"
                    }
                }
            )
            print(f"Added actual image {image_path} to GPT-4o request.")
        else:
            print(f"Could not encode image {image_path}. Proceeding with text prompt only.")
    elif os.path.splitext(image_path)[1].lower() == ".txt": # Handle dummy text files
        try:
            with open(image_path, "r", encoding="utf-8") as f_text:
                dummy_content = f.read()
            messages[0]["content"].append(
                {"type": "text", "text": f"\n\n[The following is a textual description of the image content for '{os.path.basename(image_path)}']\n{dummy_content}"} 
            )
            print(f"Added dummy text content from {image_path} to GPT-4o request.")
        except Exception as e:
            print(f"Error reading dummy text file {image_path}: {e}. Proceeding with base prompt only.")
    else:
        print(f"Warning: File {image_path} is not a recognized image or .txt dummy. Proceeding with text prompt only.")

    print(f"Sending request to GPT-4o for image: {os.path.basename(image_path)}...")
    try:
 
        completion = openai_client.chat.completions.create(
            model="gemini-2.5-pro-preview-05-06", # Or your preferred GPT-4 vision model
            messages=messages
        )
        print(completion)
        response_content = completion.choices[0].message.content
        raw_response = completion.model_dump()
        print(f"Successfully received response from GPT-4o for {os.path.basename(image_path)}.")
        return response_content.strip(), raw_response
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"OpenAI API Rate Limit Error: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error (Status {e.status_code}): {e.response}")
    except Exception as e:
        print(f"An unexpected error occurred while calling OpenAI API: {e}")
    
    return None, None

def extract_lat_lon_from_text(answer):
    matches = re.findall(r"\(([^)]+)\)", answer)
    
    if not matches:
        return None
    
    last_match = matches[-1]
    
    numbers = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", last_match)
    
    if len(numbers) >= 2:
        try:
            lat = float(numbers[0])
            lon = float(numbers[1])
            return lat, lon
        except ValueError:
            return None
    
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

#processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto")
#model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", device_map="auto", torch_dtype=torch.bfloat16)

#print("A")

llava_device = "cuda:0"

# Use a local path instead of huggingface hub
local_clip_path = "./models/clip-vit-large-patch14-336"

vision_encoder = AutoModel.from_pretrained(local_clip_path).to(llava_device)
clip_image_processor = AutoProcessor.from_pretrained(local_clip_path)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)



Province_classifier = ProvinceModel(num_classes=37)
Province_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/province_2.pth", map_location=llava_device))
Province_classifier.to(llava_device)
Province_classifier.eval()

city_classifier = CityModel(num_classes=368)
city_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/city_2.pth", map_location=llava_device))
city_classifier.to(llava_device)
city_classifier.eval()

Country_classifier = CountryModel(num_classes=2421)
Country_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/country_2.pth", map_location=llava_device))
Country_classifier.to(llava_device)
Country_classifier.eval()

#print("C")

#print("orz")
with open("/root/autodl-tmp/TUXUN/src/shape_centers.json", "r") as f:
    shape_cneter = json.load(f)
with open("/root/autodl-tmp/TUXUN/src/shape_centers_1.json", "r") as f:
    shape_cneter_1 = json.load(f)
with open("/root/autodl-tmp/TUXUN/src/shape_centers_3.json", "r") as f:
    shape_cneter_3 = json.load(f)    
full_dataset = Dataset.load_from_disk("/root/autodl-tmp/TUXUN/dataset_one")

split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']
#val_dataset = Dataset.load_from_disk("/root/autodl-tmp/TUXUN/dataset_val")
val_dataset = val_dataset.select(range(50))

embeddings = np.vstack(train_dataset['embedding']).astype('float32')

dimension = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index

index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

distances = []
geocell_distances = []
dataset_distances = []
pred_coords_list = []
gt_coords_list = []
valid_cnt = 0

pbar = tqdm(val_dataset, desc="Processing images", unit="img")

distance = [1, 25, 200, 750, 2500]
sum_dist = [0, 0, 0, 0, 0]
geocell_sum_dist = [0, 0, 0, 0, 0]
dataset_sum_dist = [0, 0, 0, 0, 0]

#ocr = PaddleOCR()
ocr = PaddleOCR(device="gpu")

for item in val_dataset:
    

#https://paddlepaddle.github.io/PaddleOCR/main/version3.x/pipeline_usage/OCR.html
# 调用gemini生成答案，并给gemini识别出的文字，前三个分层训练出的推测，以及最近邻图片结果，并允许它使用搜索
    prompt_text = f"""
You are a leading expert in Chinese geolocation research.

Now, given an image and related description from within China, your task is to determine its **precise geolocation**.

Make a **precise location prediction** within China, give coordinates in the format (latitude, longitude).
"""
    response_content,raw_response = generate_response(openai_cilent,item['image_filename'],prompt_text)
    answer = response_content
    print(answer)
    pred_coords = extract_lat_lon_from_text(answer)
# 如果没提取到，那么用本地答案
    if pred_coords is None:
        pred_coords = (0,0)

    gt_coords = float(item['latitude']), float(item['longitude'])
    dist = haversine_distance(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    print(pred_coords,dist,gt_coords)
    distances.append(dist)
    pred_coords_list.append(pred_coords)
    gt_coords_list.append(gt_coords)
    
    valid_cnt += 1
    for i in range(5):
        if dist < distance[i]:
            sum_dist[i] += 1
    pbar.set_description(f"Pred={pred_coords}, GT={gt_coords}, Dist={dist:.2f}km, Valid={valid_cnt}, {sum_dist}")
    pbar.update(1)
    # Print average distance error and geometry average after each image
    if len(distances) > 0:
        avg_dist = np.mean(distances)
        geom_avg = np.exp(np.mean(np.log(np.array(distances) + 1e-8)))  # add small value to avoid log(0)
        print(f"Current average distance error: {avg_dist:.2f} km, geometric mean: {geom_avg:.2f} km")

print(f"\nProcessed {len(distances)} images successfully.")
print(f"Average distance error: {np.mean(distances):.2f}, {np.mean(geocell_distances)}, {np.mean(dataset_distances)} km")

for i in range(5):
    sum_dist[i] /= valid_cnt
print(f'proportion: {sum_dist}')


