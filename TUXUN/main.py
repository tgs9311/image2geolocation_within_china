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
from infer import infer_model

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

openai_cilent = openai.OpenAI(api_key=api_key, base_url=base_url)


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

split_dataset = full_dataset.train_test_split(test_size=0.1, seed=514)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']
#val_dataset = Dataset.load_from_disk("/root/autodl-tmp/TUXUN/dataset_val")
val_dataset = val_dataset.select(range(10))

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
    
    gt_coords = float(item['latitude']), float(item['longitude'])
    clip_output = torch.tensor(item['embedding']).to(llava_device).reshape(1,768)
    Province_output = Province_classifier(clip_output)
    Province_top_probs, Province_top_indices = torch.topk(Province_output, k=5, dim=-1)
    city_output = city_classifier(clip_output)
    city_top_probs, city_top_indices = torch.topk(city_output, k=5, dim=-1)
    Country_output = Country_classifier(clip_output)
    Country_top_probs, Country_top_indices = torch.topk(Country_output, k=5, dim=-1)
    query_vector = clip_output.cpu().numpy()
    database_dis , indices = index.search(query_vector, 10)
    top3_city_indices = city_top_indices[0][:3].tolist()
    top3_city_names = [shape_cneter[idx] for idx in top3_city_indices]
    top3_Province_indices = Province_top_indices[0][:3].tolist()
    top3_Province_names = [shape_cneter_1[idx] for idx in top3_Province_indices]
    top3_Country_indices = Country_top_indices[0][:3].tolist()
    top3_Country_names = [shape_cneter_3[idx] for idx in top3_Province_indices]
    print(top3_Province_names,top3_city_names,top3_Country_names)
#RAG搜索
    pred_coords = [train_dataset['latitude'][indices[0][0]], train_dataset['longitude'][indices[0][0]]]
#调用PaddleOCR-v5识别图片中文字
    result = ocr.predict(item['image_filename'])
    wenzi = result[0]["rec_texts"]
    print(result[0]["rec_texts"])
#https://paddlepaddle.github.io/PaddleOCR/main/version3.x/pipeline_usage/OCR.html
# 调用gemini生成答案，并给gemini识别出的文字，前三个分层训练出的推测，以及最近邻图片结果，并允许它使用搜索
    prompt_text = f"""
You are a leading expert in Chinese geolocation research.

Now, given an image and related description from within China, your task is to determine its **precise geolocation**.

You are given the following inputs(Be very careful because the following result may not be accurate, only for reference):
1. **OCR-extracted Text**: {wenzi}
2. **Top-3 Predicted Provinces**: {top3_Province_names}
3. **Top-3 Predicted Cities**: {top3_city_names}
4. **Top-3 Predicted Countries**: {top3_Country_names}
5. **Nearest Neighbor Image Location Coordinates from RAG**: Latitude = {pred_coords[0]}, Longitude = {pred_coords[1]}
6. **Original Image**

Your analysis priority:
**Textual Information > Architectural Styles > Natural Environment & Vegetation > Street View Car Meta.**

You should focus your reasoning using the following key directions:

---

1. **Natural Environment Analysis:**
   - Assess overall topography, infer climate, and identify dominant vegetation patterns and soil types.
   - Note significant geographical features (e.g., major basins, plateaus, coastlines, snow) and their general regional implications within China.

2. **Linguistic and Textual Clues:**
   - Identify the primary script (Simplified Chinese).
   - Detect any minority scripts, Traditional Chinese, or relevant foreign languages.
   - Scrutinize visible text for:
     * Place names
     * Administrative divisions
     * Business names indicating regional specialties
     * Regional linguistic hints (e.g., character usage unique to certain areas)

3. **Architectural and Infrastructure Signatures:**
   - Analyze building style (urban/rural, North/South, ethnic influences)
   - Materials, roofs, balconies, air-conditioning/heating installations.
   - Observe roads: expressway types (G/S/X), sign design, road markings.
   - Look for cars: license plates, city bus logos, traffic flow (LHD/RHD).
   - Detect regionally specific public infrastructure (e.g., area codes on phone signs).

4. **Street View Meta Clues:**
   - Recognize the source of the street view (Baidu, Tencent, Google).
   - If visible, assess car type/generation.
   - Consider any visible landmarks for geo-inference.

---

Finally, synthesize all available information to make a **precise location prediction** within China, give coordinates in the format (latitude, longitude).
"""
    response_content,raw_response = generate_response(openai_cilent,item['image_filename'],prompt_text)
    answer = response_content
    print(answer)
    pred_coords = extract_lat_lon_from_text(answer)
    print(pred_coords)

    local_pred_coords = infer_model(img_path=item['image_filename'])

    if pred_coords is None:
        pred_coords = top3_city_names[0]['center']
    elif pred_coords[0] == 0 or pred_coords[0] == 0:
        pred_coords = top3_city_names[0]['center']

    dist = haversine_distance(pred_coords[0], pred_coords[1], gt_coords[0], gt_coords[1])
    geocell_dist = haversine_distance(top3_city_names[0]['center'][0], top3_city_names[0]['center'][1], gt_coords[0], gt_coords[1])
    dataset_dist = haversine_distance(train_dataset['latitude'][indices[0][0]], train_dataset['longitude'][indices[0][0]], gt_coords[0], gt_coords[1])
    distances.append(dist)
    geocell_distances.append(geocell_dist)
    dataset_distances.append(dataset_dist)
    pred_coords_list.append(pred_coords)
    gt_coords_list.append(gt_coords)
    
    valid_cnt += 1
    for i in range(5):
        if dist < distance[i]:
            sum_dist[i] += 1
        if geocell_dist < distance[i]:
            geocell_sum_dist[i] += 1
        if dataset_dist < distance[i]:
            dataset_sum_dist[i] += 1
    pbar.set_description(f"Pred={pred_coords}, GT={gt_coords}, Dist={dist:.2f}km, Valid={valid_cnt}, {sum_dist}, {geocell_sum_dist} ,{dataset_sum_dist}")
    pbar.update(1)

print(f"\nProcessed {len(distances)} images successfully.")
print(f"Average distance error: {np.mean(distances):.2f}, {np.mean(geocell_distances)}, {np.mean(dataset_distances)} km")

for i in range(5):
    sum_dist[i] /= valid_cnt
    geocell_sum_dist[i] /= valid_cnt
    dataset_sum_dist[i] /= valid_cnt
print(f'proportion: {sum_dist}, {geocell_sum_dist}, {dataset_sum_dist}')


