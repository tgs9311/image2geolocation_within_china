#Use picture under ./source 
import torch
import json
import re
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import math
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel
# Assuming src.model contains the definitions for CountryModel, CityModel, ProvinceModel
from src.model import CountryModel, CityModel, ProvinceModel
from torch.utils.data import DataLoader, random_split
# import matplotlib.pyplot as plt # Not used
import numpy as np
# import scipy.io # Not used
from tqdm import tqdm
import faiss
from datasets import Dataset # Still needed for train_dataset for RAG
import io
from paddleocr import PaddleOCR
import openai
from dataclasses import dataclass, asdict
import os
import random
import base64
import mimetypes
# --- Helper Functions (mostly unchanged) ---
def get_image_media_type(image_path):
    """Determines the media type of an image file."""
    mimetype, _ = mimetypes.guess_type(image_path)
    if mimetype and mimetype.startswith("image/"):
        return mimetype
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
            print(f"Added actual image {image_path} to request.")
        else:
            print(f"Could not encode image {image_path}. Proceeding with text prompt only.")
    elif os.path.splitext(image_path)[1].lower() == ".txt": # Handle dummy text files
        try:
            with open(image_path, "r", encoding="utf-8") as f_text:
                dummy_content = f_text.read() # Corrected: f to f_text
            messages[0]["content"].append(
                {"type": "text", "text": f"\n\n[The following is a textual description of the image content for '{os.path.basename(image_path)}']\n{dummy_content}"}
            )
            print(f"Added dummy text content from {image_path} to request.")
        except Exception as e:
            print(f"Error reading dummy text file {image_path}: {e}. Proceeding with base prompt only.")
    else:
        print(f"Warning: File {image_path} is not a recognized image or .txt dummy. Proceeding with text prompt only.")

    print(f"Sending request to cloud LVLM for image: {os.path.basename(image_path)}...")
    try:
        completion = openai_client.chat.completions.create(
            model="gemini-2.5-pro-preview-05-06", # Or your preferred GPT-4 vision model
            messages=messages
        )
        # print(completion) # Optional: for debugging
        response_content = completion.choices[0].message.content
        raw_response = completion.model_dump()
        print(f"Successfully received response from cloud LVLM for {os.path.basename(image_path)}.")
        return response_content.strip(), raw_response
    except openai.APIConnectionError as e:
        print(f"API Connection Error: {e}")
    except openai.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error (Status {e.status_code}): {e.response}")
    except Exception as e:
        print(f"An unexpected error occurred while calling API: {e}")
    return None, None

def extract_lat_lon_from_text(answer: Optional[str]) -> Optional[Tuple[float, float]]:
    if not answer:
        return None
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


llava_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {llava_device}")

local_clip_path = "./models/clip-vit-large-patch14-336" # Ensure this path is correct

try:
    vision_encoder = AutoModel.from_pretrained(local_clip_path).to(llava_device)
    clip_image_processor = AutoProcessor.from_pretrained(local_clip_path)
except Exception as e:
    print(f"Error loading CLIP model from {local_clip_path}: {e}")
    print("Please ensure the CLIP model files are present at the specified path.")
    exit()


# --- OpenAI Client Initialization ---
# IMPORTANT: Replace with your actual API key and base URL if necessary
# Ensure your API key has the necessary permissions for the model you are calling.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Use environment variable or direct string
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY.startswith("sk-"):
    print("Warning: OpenAI API key might be missing or invalid.")

try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None # Set to None if initialization fails


# --- Classifier Models Loading ---
# Ensure these paths are correct for your environment
try:
    Province_classifier = ProvinceModel(num_classes=37)
    Province_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/province_2.pth", map_location=llava_device))
    Province_classifier.to(llava_device)
    Province_classifier.eval()

    city_classifier = CityModel(num_classes=368)
    city_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/city_2.pth", map_location=llava_device))
    city_classifier.to(llava_device)
    city_classifier.eval()

    Country_classifier = CountryModel(num_classes=2421) # Assuming CountryModel exists
    Country_classifier.load_state_dict(torch.load("/root/autodl-tmp/TUXUN/src/country_2.pth", map_location=llava_device))
    Country_classifier.to(llava_device)
    Country_classifier.eval()
except Exception as e:
    print(f"Error loading classifier models: {e}")
    print("Please ensure model paths and definitions (CountryModel, CityModel, ProvinceModel) are correct.")
    # Depending on the desired behavior, you might want to exit or proceed without classifiers.
    # For now, we'll let it try to continue, but operations relying on classifiers will fail.


# --- Shape Centers and Dataset for RAG ---
try:
    with open("/root/autodl-tmp/TUXUN/src/shape_centers.json", "r") as f:
        shape_cneter = json.load(f) # Corrected variable name for consistency
    with open("/root/autodl-tmp/TUXUN/src/shape_centers_1.json", "r") as f:
        shape_cneter_1 = json.load(f)
    with open("/root/autodl-tmp/TUXUN/src/shape_centers_3.json", "r") as f:
        shape_cneter_3 = json.load(f)
except Exception as e:
    print(f"Error loading shape center files: {e}")
    # Handle missing shape files, perhaps by exiting or using defaults
    shape_cneter, shape_cneter_1, shape_cneter_3 = {}, {}, {}


# --- FAISS Index for RAG ---
# This still requires a dataset with 'embedding', 'latitude', 'longitude' columns
# Ensure this dataset path is correct and the dataset is available.
FAISS_INDEX_ENABLED = True
try:
    full_dataset_path = "/root/autodl-tmp/TUXUN/dataset_one"
    if os.path.exists(full_dataset_path):
        full_dataset = Dataset.load_from_disk(full_dataset_path)
        # The original script splits, we only need the 'train' part for RAG embeddings
        # If your dataset is already prepared and doesn't need splitting, adjust accordingly.
        # For simplicity, let's assume full_dataset can be used directly or is the 'train' part.
        # If you have a distinct training set for RAG embeddings, load that.
        # This example uses 'full_dataset' as if it's the RAG source.
        train_dataset_for_rag = full_dataset # Adjust if you have a specific train split for RAG
        
        embeddings = np.vstack(train_dataset_for_rag['embedding']).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"FAISS index built with {index.ntotal} vectors.")
    else:
        print(f"Warning: RAG dataset not found at {full_dataset_path}. FAISS index will not be built. RAG lookups will be skipped.")
        FAISS_INDEX_ENABLED = False
        index = None
        train_dataset_for_rag = None # Ensure it's None if not loaded

except Exception as e:
    print(f"Error loading dataset for FAISS or building index: {e}")
    FAISS_INDEX_ENABLED = False
    index = None
    train_dataset_for_rag = None


# --- OCR Initialization ---
try:
    ocr = PaddleOCR(device="gpu")
except Exception as e:
    print(f"Error initializing PaddleOCR: {e}. OCR will not be available.")
    ocr = None


# --- Image Processing Loop ---
source_image_dir = Path("./source")
if not source_image_dir.exists() or not source_image_dir.is_dir():
    print(f"Source directory {source_image_dir} not found. Please create it and add images.")
    exit()

image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
image_files = []
for ext in image_extensions:
    image_files.extend(list(source_image_dir.glob(ext)))

if not image_files:
    print(f"No images found in {source_image_dir}.")
    exit()

results_output = [] # To store final results

pbar = tqdm(image_files, desc="Processing images", unit="img")

for image_path in pbar:
    print(f"\n--- Processing: {image_path.name} ---")
    final_pred_coords = None

    try:
        # 1. Load image and get CLIP embedding
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = clip_image_processor(images=image, return_tensors="pt")
            pixel_values = inputs.get('pixel_values')

            if pixel_values is None:
                raise ValueError("pixel_values not found in clip_image_processor output.")
            
            pixel_values = pixel_values.to(llava_device) # Move to device

            with torch.no_grad():
                # Use get_image_features() for CLIPModel, which is what AutoModel usually loads for CLIP.
                # This directly gives the image embeddings (equivalent to the CLS token's representation).
                image_features = vision_encoder.get_image_features(pixel_values=pixel_values)
            
            # image_features is typically of shape [batch_size, embedding_dim]
            # For a single image, this will be [1, embedding_dim]
            clip_output = image_features.reshape(1, -1) # Ensures (1, embedding_dim)

        except Exception as e:
            print(f"Error processing image {image_path.name} for CLIP: {e}")
            clip_output = None

        # Initialize predicted names and RAG coords with defaults
        top3_Province_names_str = "Unavailable"
        top3_city_names_str = "Unavailable"
        top3_Country_names_str = "Unavailable" # Corrected variable name
        rag_pred_coords_str = "Unavailable"
        extracted_text_str = "Unavailable"
        query_vector = clip_output.cpu().numpy()
        database_dis , indices = index.search(query_vector, 10)
        rag_pred_coords_str ="Latitude ="+str(train_dataset_for_rag['latitude'][indices[0][0]])+",Longitude ="+str(train_dataset_for_rag['longitude'][indices[0][0]]);
        print(rag_pred_coords_str)
        if clip_output is not None:
            # 2. Classifier predictions
            # Initialize to safe defaults that match the types expected by the rest of your script
            top3_Province_names = [{"name": "Province Info Unavailable"}] # List of objects (dictionaries)
            top3_city_names = [{"name": "City Info Unavailable", "center": [0.0, 0.0]}] # List of objects, used for fallback
            top3_Country_names = [{"name": "Country Info Unavailable"}] # List of objects

            # These _str versions are used in your prompt
            top3_Province_names_str = ["Province Info Unavailable"]
            top3_city_names_str = ["City Info Unavailable"]
            top3_Country_names_str = ["Country Info Unavailable"]
            
            try:
                # This 'with' block and the model calls are from your provided script
                with torch.no_grad():
                    Province_output_tensor = Province_classifier(clip_output) # Renamed to avoid conflict
                    Province_top_probs, Province_top_indices = torch.topk(Province_output_tensor, k=3, dim=-1)
                    
                    city_output_tensor = city_classifier(clip_output) # Renamed
                    city_top_probs, city_top_indices = torch.topk(city_output_tensor, k=3, dim=-1)
                    
                    Country_output_tensor = Country_classifier(clip_output) # Renamed
                    Country_top_probs, Country_top_indices_tensor = torch.topk(Country_output_tensor, k=3, dim=-1) # Renamed

                # --- Retrieve objects using direct indexing (user's preferred style) ---
                # Assuming shape_cneter_1, shape_cneter, shape_cneter_3 are LISTS
                # and the indices from torch.topk are valid for these lists.

                province_indices_list = Province_top_indices[0][:3].tolist()
                default_prov_obj = {"name": "Unknown Province (Index Error)"} # Fallback for out-of-bounds
                # top3_Province_names will be a list of objects (e.g., dictionaries)
                top3_Province_names = [
                    shape_cneter_1[idx] if 0 <= idx < len(shape_cneter_1) else default_prov_obj
                    for idx in province_indices_list
                ]

                city_indices_list = city_top_indices[0][:3].tolist()
                default_city_obj = {"name": "Unknown City (Index Error)", "center": [0.0, 0.0]}
                # top3_city_names will be a list of objects, used for fallback logic later
                top3_city_names = [
                    shape_cneter[idx] if 0 <= idx < len(shape_cneter) else default_city_obj
                    for idx in city_indices_list
                ]
                
                country_indices_list = Country_top_indices_tensor[0][:3].tolist() # Use the correctly named indices
                default_country_obj = {"name": "Unknown Country/District (Index Error)"}
                # top3_Country_names will be a list of objects
                top3_Country_names = [
                    shape_cneter_3[idx] if 0 <= idx < len(shape_cneter_3) else default_country_obj
                    for idx in country_indices_list # Corrected: was Country_top_indices in your original script's example
                ]

                # --- Create string versions for the prompt (matching your script's variable names) ---
                # These attempt to get a 'name' field, otherwise stringify the object.
                top3_Province_names_str = [obj.get('shapeName', obj.get('name', "Unknown Province")) for obj in top3_Province_names]
                top3_city_names_str = [obj.get('shapeName', obj.get('name', "Unknown Province")) for obj in top3_city_names]
                top3_Country_names_str = [obj.get('shapeName', obj.get('name', "Unknown Province")) for obj in top3_country_names]
                
                # This print statement is from your script, using the _str versions
                print(f"Top Provinces: {top3_Province_names_str}, Top Cities: {top3_city_names_str}, Top Districts/Counties: {top3_Country_names_str}")

            except Exception as e:
                print(f"Error during classifier prediction for {image_path.name}: {e}")
                # If an error occurs, the pre-initialized safe default values for 
                # top3_Province_names, top3_city_names, top3_Country_names,
                # and their _str versions will be used by the rest of the script.


        # 4. PaddleOCR
        wenzi = "OCR Unavailable"
        if ocr:
            try:
                ocr_result = ocr.predict(str(image_path)) # Use ocr.ocr for combined detection and recognition
                if ocr_result and ocr_result[0]: # Check if result is not None and not empty
                    wenzi = ocr_result[0]["rec_texts"]
                else:
                    wenzi = "No text detected"
                #extracted_text_str = "; ".join(wenzi_list)
                print(f"OCR Extracted Text: {wenzi}")
            except Exception as e:
                print(f"Error during OCR for {image_path.name}: {e}")
                wenzi = "OCR Error"
        else:
            wenzi = "OCR Not Initialized"
    

        # 5. Call Gemini/GPT-4o
        prompt_text = f"""
You are a leading expert in Chinese geolocation research.

Now, given an image and related description from within China, your task is to determine its **precise geolocation**.

You are given the following inputs (Be very careful because the following result may not be accurate, only for reference):
1. **OCR-extracted Text**: {wenzi}
2. **Top-3 Predicted Provinces**: {top3_Province_names_str}
3. **Top-3 Predicted Cities**: {top3_city_names_str}
4. **Top-3 Predicted Districts/Counties**: {top3_Country_names_str}
5. **Nearest Neighbor Image Location Coordinates from RAG**: {rag_pred_coords_str}
6. **Original Image**

Your analysis priority:
**Textual Information > Architectural Styles > Natural Environment & Vegetation > Street View Car Meta.**
You can use search to help or verify your reasoning.
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
        llm_response_content, raw_llm_response = None, None
        if openai_client:
            llm_response_content, raw_llm_response = generate_response(openai_client, str(image_path), prompt_text)
        else:
            print("client not available, skipping LLM call.")


        if llm_response_content:
            print(f"LLM Response for {image_path.name}:\n{llm_response_content}")
            final_pred_coords = extract_lat_lon_from_text(llm_response_content)
        else:
            print(f"No response from LLM for {image_path.name}.")

        # Fallback if LLM fails or doesn't provide coordinates
        if final_pred_coords is None or (final_pred_coords[0] == 0 and final_pred_coords[1] == 0): # Also check for (0,0) as potential failure
            print(f"LLM did not provide valid coordinates for {image_path.name}. Using fallback from top predicted city.")
            if top3_city_names and 'center' in top3_city_names[0] and isinstance(top3_city_names[0]['center'], list) and len(top3_city_names[0]['center']) == 2:
                final_pred_coords = tuple(top3_city_names[0]['center'])
            else:
                print(f"Fallback city center for {image_path.name} is also unavailable or malformed. Setting to (0,0).")
                final_pred_coords = (0.0, 0.0) # Ultimate fallback

        print(f"==> Final Predicted Coordinates for {image_path.name}: {final_pred_coords} <==")
        results_output.append({
            "image": image_path.name,
            "predicted_latitude": final_pred_coords[0] if final_pred_coords else None,
            "predicted_longitude": final_pred_coords[1] if final_pred_coords else None,
            "llm_full_response": llm_response_content
        })

    except Exception as e:
        print(f"An unexpected error occurred processing {image_path.name}: {e}")
        results_output.append({
            "image": image_path.name,
            "predicted_latitude": None,
            "predicted_longitude": None,
            "error": str(e)
        })
    finally:
        pbar.set_description(f"Processed {image_path.name}, Last Pred: {final_pred_coords}")
        pbar.update(1)

pbar.close()

print("\n--- Geolocation Processing Complete ---")
print("Final Results:")
for res in results_output:
    if "error" in res:
        print(f"Image: {res['image']}, Error: {res['error']}")
    else:
        print(f"Image: {res['image']}, Predicted Coords: (Lat: {res['predicted_latitude']:.4f}, Lon: {res['predicted_longitude']:.4f})")
        # print(f"LLM Response: {res['llm_full_response'][:200]}...") # Optionally print part of the LLM response

# Save results to a JSON file
output_file_path = "./geolocation_results.json"
try:
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        json.dump(results_output, f_out, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {output_file_path}")
except Exception as e:
    print(f"Error saving results to JSON: {e}")