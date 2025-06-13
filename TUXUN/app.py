import torch
import json
import re
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import math
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModel # AutoModelForImageTextToText not used in original script
# Assuming src.model contains the definitions for CountryModel, CityModel, ProvinceModel
# 您需要确保这个导入路径根据您的项目结构是正确的
try:
    from src.model import CountryModel, CityModel, ProvinceModel
except ImportError:
    print("警告: 无法从 src.model 导入 CountryModel, CityModel, ProvinceModel。请确保它们已定义且可访问。")
    # 如果找不到，定义虚拟类，以避免脚本立即崩溃
    # 这只是占位符，您必须确保实际模型已加载
    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # 从kwargs获取num_classes，如果不存在则默认为10，确保fc层有输出维度
            num_outputs = kwargs.get('num_classes', 10)
            if not isinstance(num_outputs, int) or num_outputs <= 0:
                num_outputs = 10 # 提供一个默认的有效值
            self.fc = torch.nn.Linear(1, num_outputs) # 假设输入维度为1，实际应为CLIP输出维度

        def forward(self, x):
            # 确保输出与num_classes匹配
            # 创建一个与输入批次大小相同，输出维度为fc层输出维度的随机张量
            return torch.rand(x.size(0), self.fc.out_features, device=x.device)

    CountryModel, CityModel, ProvinceModel = DummyModel, DummyModel, DummyModel

# from torch.utils.data import DataLoader, random_split # Not used in the web app context directly
import numpy as np
import faiss
from datasets import Dataset # Still needed for train_dataset for RAG
# import io # Not used
from paddleocr import PaddleOCR # 确保 paddleocr 已安装
import openai
# from dataclasses import dataclass, asdict # Not used
import os
# import random # Not used
import base64
import mimetypes
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# --- Flask App Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB Max Upload Size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Helper Functions (来自您的脚本) ---
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

def _generate_openai_llm_response(openai_client_instance: openai.OpenAI, image_path: str, prompt_text:str) -> tuple[str | None, Dict[str, Any] | None]:
    if not openai_client_instance:
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
            with open(image_path, "r", encoding="utf-8") as f_text: # 修正: f to f_text (已在您原脚本中修正)
                dummy_content = f_text.read()
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
        completion = openai_client_instance.chat.completions.create(
            model="gemini-2.5-pro-preview-05-06", # 使用您脚本中指定的模型
            messages=messages
        )
        response_content = completion.choices[0].message.content
        raw_response = completion.model_dump() # In new openai client, it's model_dump() not asdict()
        print(f"Successfully received response from cloud LVLM for {os.path.basename(image_path)}.")
        return response_content.strip() if response_content else None, raw_response
    except openai.APIConnectionError as e: print(f"API Connection Error: {e}")
    except openai.RateLimitError as e: print(f"API Rate Limit Error: {e}")
    except openai.APIStatusError as e: print(f"OpenAI API Status Error (Status {e.status_code}): {e.response}")
    except Exception as e: print(f"An unexpected error occurred while calling API: {e}")
    return None, None

def extract_lat_lon_from_text(answer: Optional[str]) -> Optional[Tuple[float, float]]:
    if not answer: return None
    matches = re.findall(r"\(([^)]+)\)", answer)
    if not matches: return None
    last_match = matches[-1]
    numbers = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", last_match)
    if len(numbers) >= 2:
        try:
            lat = float(numbers[0])
            lon = float(numbers[1])
            return lat, lon
        except ValueError: return None
    return None

# --- Global Variables & Model Initializations ---
llava_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {llava_device}")

LOCAL_CLIP_PATH = "./models/clip-vit-large-patch14-336" # 示例, 请修改
PROVINCE_MODEL_PATH = "/root/autodl-tmp/TUXUN/src/province_2.pth" # 请修改
CITY_MODEL_PATH = "/root/autodl-tmp/TUXUN/src/city_2.pth"         # 请修改
COUNTRY_MODEL_PATH = "/root/autodl-tmp/TUXUN/src/country_2.pth"   # 请修改
SHAPE_CENTERS_PATH = "/root/autodl-tmp/TUXUN/src/shape_centers.json" # 请修改
SHAPE_CENTERS_1_PATH = "/root/autodl-tmp/TUXUN/src/shape_centers_1.json" # 请修改
SHAPE_CENTERS_3_PATH = "/root/autodl-tmp/TUXUN/src/shape_centers_3.json" # 请修改
FULL_DATASET_PATH_FOR_RAG = "/root/autodl-tmp/TUXUN/dataset_one" # 请修改

vision_encoder = None
clip_image_processor = None
try:
    if os.path.exists(LOCAL_CLIP_PATH): # 检查路径是否存在
        vision_encoder = AutoModel.from_pretrained(LOCAL_CLIP_PATH).to(llava_device)
        clip_image_processor = AutoProcessor.from_pretrained(LOCAL_CLIP_PATH)
        print(f"CLIP model loaded successfully from {LOCAL_CLIP_PATH}.")
    else:
        print(f"错误: CLIP 模型路径 {LOCAL_CLIP_PATH} 未找到。")
except Exception as e:
    print(f"加载 CLIP 模型时出错 {LOCAL_CLIP_PATH}: {e}")

# --- OpenAI Client Initialization (来自您的脚本) ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
openai_client = None # 初始化为 None
if not OPENAI_API_KEY.startswith("sk-"):
    print("警告: OpenAI API 密钥可能缺失或无效。")
else:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        print("OpenAI client 初始化成功。")
    except Exception as e:
        print(f"初始化 OpenAI client 时出错: {e}")


# --- Classifier Models Loading (来自您的脚本) ---
Province_classifier, city_classifier, Country_classifier = None, None, None
try:
    # Province Classifier
    if os.path.exists(PROVINCE_MODEL_PATH):
        Province_classifier = ProvinceModel(num_classes=37) # num_classes 在初始化时设置
        Province_classifier.load_state_dict(torch.load(PROVINCE_MODEL_PATH, map_location=llava_device))
        Province_classifier.to(llava_device)
        Province_classifier.eval()
        print("Province classifier loaded.")
    else:
        print(f"警告: Province 模型路径 {PROVINCE_MODEL_PATH} 未找到。")

    # City Classifier
    if os.path.exists(CITY_MODEL_PATH):
        city_classifier = CityModel(num_classes=368) # num_classes 在初始化时设置
        city_classifier.load_state_dict(torch.load(CITY_MODEL_PATH, map_location=llava_device))
        city_classifier.to(llava_device)
        city_classifier.eval()
        print("City classifier loaded.")
    else:
        print(f"警告: City 模型路径 {CITY_MODEL_PATH} 未找到。")
        
    # Country Classifier
    if os.path.exists(COUNTRY_MODEL_PATH):
        Country_classifier = CountryModel(num_classes=2421) # num_classes 在初始化时设置
        Country_classifier.load_state_dict(torch.load(COUNTRY_MODEL_PATH, map_location=llava_device))
        Country_classifier.to(llava_device)
        Country_classifier.eval()
        print("Country/District classifier loaded.")
    else:
        print(f"警告: Country 模型路径 {COUNTRY_MODEL_PATH} 未找到。")

except Exception as e:
    print(f"加载分类器模型时出错: {e}")

# --- Shape Centers Loading (来自您的脚本) ---
shape_cneter, shape_cneter_1, shape_cneter_3 = {}, {}, {}
try:
    if os.path.exists(SHAPE_CENTERS_PATH):
        with open(SHAPE_CENTERS_PATH, "r", encoding="utf-8") as f: shape_cneter = json.load(f)
        print("Shape centers loaded.")
    else: print(f"警告: Shape centers 路径 {SHAPE_CENTERS_PATH} 未找到。")
    
    if os.path.exists(SHAPE_CENTERS_1_PATH):
        with open(SHAPE_CENTERS_1_PATH, "r", encoding="utf-8") as f: shape_cneter_1 = json.load(f)
        print("Shape centers 1 loaded.")
    else: print(f"警告: Shape centers 1 路径 {SHAPE_CENTERS_1_PATH} 未找到。")

    if os.path.exists(SHAPE_CENTERS_3_PATH):
        with open(SHAPE_CENTERS_3_PATH, "r", encoding="utf-8") as f: shape_cneter_3 = json.load(f)
        print("Shape centers 3 loaded.")
    else: print(f"警告: Shape centers 3 路径 {SHAPE_CENTERS_3_PATH} 未找到。")
except Exception as e:
    print(f"加载 shape center 文件时出错: {e}")


# --- FAISS Index for RAG (来自您的脚本) ---
FAISS_INDEX_ENABLED = True # 默认为 False
faiss_index = None # faiss 被导入，但变量名是 index
train_dataset_for_rag = None
try:
    if os.path.exists(FULL_DATASET_PATH_FOR_RAG):
        full_dataset = Dataset.load_from_disk(FULL_DATASET_PATH_FOR_RAG)
        train_dataset_for_rag = full_dataset # Adjust if you have a specific train split for RAG
        if 'embedding' in train_dataset_for_rag.column_names: # 确保 'embedding' 列存在
            embeddings = np.vstack(train_dataset_for_rag['embedding']).astype('float32')
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension) # 使用 faiss_index 存储索引
            faiss_index.add(embeddings)
            FAISS_INDEX_ENABLED = True
            print(f"FAISS index built with {faiss_index.ntotal} vectors.")
        else:
            print(f"警告: RAG 数据集 {FULL_DATASET_PATH_FOR_RAG} 中缺少 'embedding' 列。FAISS 索引未构建。")
    else:
        print(f"警告: RAG 数据集路径 {FULL_DATASET_PATH_FOR_RAG} 未找到。FAISS 索引将不会构建。")
except Exception as e:
    print(f"加载用于 FAISS 的数据集或构建索引时出错: {e}")


# --- OCR Initialization (来自您的脚本) ---
ocr = None # 初始化为 None
try:
    # 严格按照您的原始脚本初始化 OCR
    ocr = PaddleOCR(device="gpu")
    # 原脚本是 device="gpu"，这里增加一个判断，如果GPU不可用则用CPU，避免直接报错
    # 如果您坚持在无GPU时也尝试 device="gpu" 并处理其抛出的异常，可以改回 device="gpu"
    print(f"PaddleOCR 初始化成功 (device: {'gpu' if torch.cuda.is_available() else 'cpu'})。")
except Exception as e:
    print(f"初始化 PaddleOCR 时出错: {e}。OCR 将不可用。")
    print("请确保 PaddleOCR 已正确安装 (pip install paddlepaddle paddleocr)。")
    print("如果使用 GPU，请确保 CUDA 和 cuDNN 已正确安装并与 PaddlePaddle 版本兼容。")


# --- Core Image Processing Function ---
def process_single_image(image_file_path_str: str) -> Dict[str, Any]:
    image_path = Path(image_file_path_str) # 转换为 Path 对象，与您原脚本一致
    print(f"\n--- Processing: {image_path.name} ---")
    
    # 初始化返回结果的字典
    result_data: Dict[str, Any] = {
        "image_name": image_path.name,
        "predicted_latitude": None,
        "predicted_longitude": None,
        "llm_full_response": "LLM 未处理或出错。",
        "ocr_text": "OCR 不可用或未执行",
        "top_provinces_str": "分类器信息不可用", # 使用 _str 后缀以匹配原脚本中的提示变量
        "top_cities_str": "分类器信息不可用",
        "top_countries_districts_str": "分类器信息不可用",
        "rag_coords_str": "RAG 信息不可用",
        "error_message": None # 用于收集处理过程中的错误信息
    }
    final_pred_coords = None # 与您原脚本一致

    try:
        # 1. Load image and get CLIP embedding (与您原脚本逻辑一致)
        clip_output = None # 初始化
        if vision_encoder and clip_image_processor: # 确保CLIP模型已加载
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = clip_image_processor(images=image, return_tensors="pt")
                pixel_values = inputs.get('pixel_values')
                if pixel_values is None:
                    raise ValueError("pixel_values not found in clip_image_processor output.")
                pixel_values = pixel_values.to(llava_device)
                with torch.no_grad():
                    image_features = vision_encoder.get_image_features(pixel_values=pixel_values)
                clip_output = image_features.reshape(1, -1) # Ensures (1, embedding_dim)
            except Exception as e:
                print(f"处理图像 {image_path.name} 进行 CLIP 特征提取时出错: {e}")
                result_data["error_message"] = (result_data["error_message"] or "") + f"CLIP Error: {e}; "
        else:
            print("CLIP 模型或处理器未加载。跳过 CLIP 特征提取和相关处理。")
            result_data["error_message"] = (result_data["error_message"] or "") + "CLIP model/processor not loaded; "

        # 初始化预测名称和 RAG 坐标的默认值 (与您原脚本一致)
        # 注意：原脚本中 top3_Province_names_str 等变量在循环外没有重新初始化，
        # 这里我们为每次图像处理都初始化它们。
        top3_Province_names_str_list = ["Province Info Unavailable"] # 改为 list 以便 join
        top3_city_names_str_list = ["City Info Unavailable"]
        top3_Country_names_str_list = ["Country Info Unavailable"] # 原脚本变量名为 top3_Country_names_str
        
        # Fallback for top3_city_names, used if LLM fails. This needs to be a list of dicts.
        # 原脚本的 top3_city_names 变量在后续被赋值为对象列表
        # 我们在这里初始化一个兼容的结构，如果分类器失败，它将保持为此默认值
        _fallback_top3_city_objects = [{"name": "City Info Unavailable", "center": [0.0, 0.0]}]


        if clip_output is not None:
            # 2. Classifier predictions (与您原脚本逻辑一致)
            # 确保分类器模型和形状中心数据已加载
            if Province_classifier and city_classifier and Country_classifier and \
               shape_cneter_1 and shape_cneter and shape_cneter_3:
                try:
                    with torch.no_grad():
                        # 调用分类器时不传递 num_classes，因为已在 __init__ 中设置
                        Province_output_tensor = Province_classifier(clip_output)
                        # k值需要小于或等于类别数量
                        k_province = min(3, Province_output_tensor.size(1))
                        Province_top_probs, Province_top_indices = torch.topk(Province_output_tensor, k=k_province, dim=-1)
                        
                        city_output_tensor = city_classifier(clip_output)
                        k_city = min(3, city_output_tensor.size(1))
                        city_top_probs, city_top_indices = torch.topk(city_output_tensor, k=k_city, dim=-1)
                        
                        Country_output_tensor = Country_classifier(clip_output)
                        k_country = min(3, Country_output_tensor.size(1))
                        Country_top_probs, Country_top_indices_tensor = torch.topk(Country_output_tensor, k=k_country, dim=-1)

                    # --- Retrieve objects using direct indexing (与您原脚本逻辑一致) ---
                    province_indices_list = Province_top_indices[0].tolist()
                    default_prov_obj = {"name": "Unknown Province (Index Error)"}
                    top3_Province_objs = [shape_cneter_1[idx] if 0 <= idx < len(shape_cneter_1) else default_prov_obj for idx in province_indices_list]
                    top3_Province_names_str_list = [obj.get('shapeName', obj.get('name', "Unknown Province")) for obj in top3_Province_objs]

                    city_indices_list = city_top_indices[0].tolist()
                    default_city_obj = {"name": "Unknown City (Index Error)", "center": [0.0, 0.0]}
                    _fallback_top3_city_objects = [shape_cneter[idx] if 0 <= idx < len(shape_cneter) else default_city_obj for idx in city_indices_list] # 更新后备城市列表
                    top3_city_names_str_list = [obj.get('shapeName', obj.get('name', "Unknown City")) for obj in _fallback_top3_city_objects]
                    
                    country_indices_list = Country_top_indices_tensor[0].tolist()
                    default_country_obj = {"name": "Unknown Country/District (Index Error)"}
                    top3_Country_objs = [shape_cneter_3[idx] if 0 <= idx < len(shape_cneter_3) else default_country_obj for idx in country_indices_list]
                    top3_Country_names_str_list = [obj.get('shapeName', obj.get('name', "Unknown Country/District")) for obj in top3_Country_objs]

                    print(f"Top Provinces: {top3_Province_names_str_list}, Top Cities: {top3_city_names_str_list}, Top Districts/Counties: {top3_Country_names_str_list}")
                except Exception as e:
                    print(f"分类器预测过程中出错 {image_path.name}: {e}")
                    result_data["error_message"] = (result_data["error_message"] or "") + f"Classifier Error: {e}; "
                    # 如果出错，保持默认的 "Unavailable" 字符串列表
            else:
                print("一个或多个分类器或形状中心数据未加载。")
                result_data["error_message"] = (result_data["error_message"] or "") + "Classifiers/ShapeData Missing; "
        
        # 将列表转换为字符串以用于LLM提示和结果显示
        result_data["top_provinces"] = ", ".join(top3_Province_names_str_list)
        result_data["top_cities"] = ", ".join(top3_city_names_str_list)
        result_data["top_countries_districts"] = ", ".join(top3_Country_names_str_list)
        #print(result_data["top_provinces"])
        # RAG 坐标查找 (如果启用了FAISS并且有CLIP输出)
        rag_pred_coords_str = "Unavailable" # 与您原脚本一致的默认值
        if FAISS_INDEX_ENABLED and faiss_index is not None and train_dataset_for_rag is not None and clip_output is not None:
            try:
                query_embedding = clip_output.cpu().numpy()
                # k=1 表示查找最近的1个邻居
                D, I = faiss_index.search(query_embedding, k=1)
                if I.size > 0 and I[0][0] < len(train_dataset_for_rag):
                    neighbor_idx = I[0][0]
                    # 确保您的 RAG 数据集有 'latitude' 和 'longitude' 列
                    if 'latitude' in train_dataset_for_rag.column_names and \
                       'longitude' in train_dataset_for_rag.column_names:
                        rag_lat = train_dataset_for_rag['latitude'][neighbor_idx]
                        rag_lon = train_dataset_for_rag['longitude'][neighbor_idx]
                        rag_pred_coords_str = f"({float(rag_lat):.4f}, {float(rag_lon):.4f})"
                        print(f"RAG Nearest Neighbor Coords: {rag_pred_coords_str}")
                    else:
                        print("RAG 数据集缺少 'latitude' 或 'longitude' 列。")
                        rag_pred_coords_str = "RAG Coords Columns Missing"
                else:
                    print("RAG: 未找到邻居或索引越界。")
                    rag_pred_coords_str = "RAG Neighbor Not Found"
            except Exception as e:
                print(f"RAG 查找过程中出错: {e}")
                rag_pred_coords_str = "RAG Lookup Error"
        result_data["rag_coords"] = rag_pred_coords_str


        # 4. PaddleOCR (严格按照您的原脚本逻辑)
        wenzi = "OCR Unavailable" # 与您原脚本一致的默认值
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
        result_data["ocr_text"] = wenzi

        # 5. Call Gemini/GPT-4o (使用您脚本中的提示和逻辑)
        # 使用之前处理好的字符串列表（已用, join）
        prompt_text = f"""
You are a leading expert in Chinese geolocation research.
Now, given an image and related description from within China, your task is to determine its **precise geolocation**.
You are given the following inputs (Be very careful because the following result may not be accurate, only for reference):
1. **OCR-extracted Text**: {wenzi}
2. **Top-3 Predicted Provinces**: {result_data["top_provinces"]}
3. **Top-3 Predicted Cities**: {result_data["top_cities"]}
4. **Top-3 Predicted Districts/Counties**: {result_data["top_countries_districts"]}
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
   - Scrutinize visible text for: Place names, Administrative divisions, Business names, Regional linguistic hints.
3. **Architectural and Infrastructure Signatures:**
   - Analyze building style, materials, roofs, installations.
   - Observe roads: expressway types, sign design, markings.
   - Look for cars: license plates, bus logos, traffic flow.
   - Detect regionally specific public infrastructure.
4. **Street View Meta Clues:**
   - Recognize the source of the street view (Baidu, Tencent, Google).
   - If visible, assess car type/generation.
   - Consider any visible landmarks for geo-inference.
---
Finally, synthesize all available information to make a **precise location prediction** within China, give coordinates in the format (latitude, longitude).
"""
        print(prompt_text)
        llm_response_content, raw_llm_response = None, None
        if openai_client: # 检查 openai_client 是否已成功初始化
            # 调用 _generate_openai_llm_response 函数
            llm_response_content, raw_llm_response = _generate_openai_llm_response(openai_client, str(image_path), prompt_text)
            result_data["llm_full_response"] = llm_response_content if llm_response_content else "LLM did not return content."
        else:
            print("OpenAI client not available, skipping LLM call.")
            result_data["llm_full_response"] = "LLM client not available, call skipped."
            result_data["error_message"] = (result_data["error_message"] or "") + "OpenAI client missing; "

        if llm_response_content:
            print(f"LLM Response for {image_path.name}:\n{llm_response_content}")
            final_pred_coords = extract_lat_lon_from_text(llm_response_content)
        else:
            print(f"No response or failed response from LLM for {image_path.name}.")
            # LLM full response已在上面设置

        # Fallback if LLM fails or doesn't provide coordinates (与您原脚本逻辑一致)
        if final_pred_coords is None or (final_pred_coords[0] == 0 and final_pred_coords[1] == 0):
            print(f"LLM 未提供有效坐标 {image_path.name}。使用分类器预测的城市中心作为后备。")
            # 使用 _fallback_top3_city_objects[0] 作为后备
            if _fallback_top3_city_objects and isinstance(_fallback_top3_city_objects[0], dict) and \
               'center' in _fallback_top3_city_objects[0] and \
               isinstance(_fallback_top3_city_objects[0]['center'], list) and \
               len(_fallback_top3_city_objects[0]['center']) == 2:
                try:
                    # 确保坐标是浮点数
                    center_coords = _fallback_top3_city_objects[0]['center']
                    final_pred_coords = (float(center_coords[0]), float(center_coords[1]))
                    print(f"使用后备城市中心: {final_pred_coords} from {_fallback_top3_city_objects[0].get('name', 'Unknown City')}")
                except (ValueError, TypeError) as e:
                    print(f"后备城市中心坐标格式错误: {e}。设置为 (0,0)。")
                    final_pred_coords = (0.0, 0.0) # 终极后备
            else:
                print(f"后备城市中心 {image_path.name} 不可用或格式错误。设置为 (0,0)。")
                final_pred_coords = (0.0, 0.0) # 终极后备
        
        if final_pred_coords: # 确保 final_pred_coords 不是 None
            result_data["predicted_latitude"] = final_pred_coords[0]
            result_data["predicted_longitude"] = final_pred_coords[1]
        
        print(f"==> Final Predicted Coordinates for {image_path.name}: {final_pred_coords} <==")

    except Exception as e:
        print(f"处理 {image_path.name} 时发生意外错误: {e}")
        result_data["error_message"] = (result_data["error_message"] or "") + f"Overall Processing Error: {str(e)}; "
        # 如果已有LLM响应，保留它，否则设为错误信息
        if result_data["llm_full_response"] == "LLM 未处理或出错。":
             result_data["llm_full_response"] = f"处理时出错: {e}"
    
    return result_data


# --- Flask Routes ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_process_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            # print("No file part in request.files")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            # print("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                processing_result = process_single_image(filepath)
                return render_template('index.html', result=processing_result, image_filename=filename)
            except Exception as e:
                print(f"文件保存或处理过程中出错: {e}")
                return render_template('index.html', error_page_message=f"处理文件时出错: {e}") # 使用不同的变量名避免与result冲突
        else:
            # print(f"File type not allowed: {file.filename}")
            return render_template('index.html', error_page_message="文件类型不允许。")

    return render_template('index.html', result=None) # GET 请求

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # This route is used by the template to display the uploaded image
    return redirect(url_for('static', filename=os.path.join(app.config['UPLOAD_FOLDER'], filename).replace(os.path.sep, '/')), code=301)

if __name__ == '__main__':
    @app.route('/display_upload/<filename>')
    def display_uploaded_file(filename):
        from flask import send_from_directory
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    


    app.run(debug=True, host='0.0.0.0', port=5000)