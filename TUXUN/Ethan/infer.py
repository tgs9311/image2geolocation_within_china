from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
import torch
import os
import json
from tqdm import tqdm
import time
import re

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

def infer_model(model_path="/root/autodl-tmp/Ethan/Qwen-VL/Qwen-VL-Models/qwen/Qwen-VL-Chat",img_path="/root/autodl-tmp/Ethan/test.jpeg"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    
    lora_weights = "/root/autodl-tmp/Ethan/Qwen-VL/LoRA/output"
    model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.bfloat16)

    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)


    query = tokenizer.from_list_format([
                {'image': img_path}, 



                {'text': "Analyze the provided image and determine its geolocation. Explain your reasoning step-by-step, as if you are an expert geolocator performing a Chain-of-Thought deduction.Finally, synthesize all available information to make a **precise location prediction**, give **coordinates in the format (latitude, longitude)** at the end."
                }
                ])
    
    response, history = model.chat(tokenizer, query=query, history=None)
    print(img_path, response)
    print(extract_lat_lon_from_text(response))
    return extract_lat_lon_from_text(response)

if __name__ == "__main__":
    infer_model()