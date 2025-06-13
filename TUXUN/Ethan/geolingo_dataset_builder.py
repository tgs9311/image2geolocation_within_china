import json
import os
import random
import base64
import mimetypes
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# Attempt to import openai and handle if not installed
try:
    import openai
except ImportError:
    print("OpenAI library not found. Please install it with: pip install openai")
    print("This script requires the OpenAI library to generate CoT descriptions with GPT-4o.")
    openai = None # Set to None so later checks can skip API calls

# Define the data structure for each item in the dataset
@dataclass
class GeoLingoDataItem:
    image_path: str
    latitude: float
    longitude: float
    cot_description: str
    raw_model_response: Dict[str, Any] = None # To store the full API response for debugging


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

def generate_cot_with_gpt4o(openai_client: openai.OpenAI, image_path: str, latitude: float, longitude: float) -> tuple[str | None, Dict[str, Any] | None]:
    """
    Generates a Chain-of-Thought (CoT) description for an image using GPT-4o,
    based on its metadata and the GeoReasoner methodology's CoT prompt.

    Args:
        openai_client (openai.OpenAI): Initialized OpenAI client.
        image_path (str): The path to the image file (can be actual image or .txt dummy).
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.

    Returns:
        tuple[str | None, Dict[str, Any] | None]: The generated CoT description string and the raw API response, or (None, None) if an error occurs.
    """
    if not openai_client:
        print("OpenAI client is not initialized. Skipping GPT-4o call.")
        return "Skipped GPT-4o call due to missing OpenAI client.", None

    # Construct the prompt based on the GeoReasoner methodology
    prompt_text = (
    f"You are a leading expert in Chinese geolocation research, specializing in the '图寻中国' (Geoguessr China) dataset. "
    f"You are given a street-view image or its description from within China (including mainland, Hong Kong, Macau, or Taiwan as covered by the dataset).\n\n"
    
    f"The **true geographic coordinates** of the image are:\n"
    f"Latitude: {latitude:.4f}, Longitude: {longitude:.4f}\n\n"

    f"Your task is to analyze the image or its description in reverse: "
    f"examine all available clues and explain how an expert would deduce this exact location, using only visual and contextual indicators.\n\n"

    f"Prioritize your analysis using the following order of clues:\n"
    f"**Textual Information > Architectural Styles > Natural Environment & Vegetation > Street View Car Meta**\n\n"

    f"Focus on these four key analytical directions:\n\n"

    f"1. **Natural Environment Analysis:**\n"
    f"   - Assess topography, infer climate zone, and identify dominant vegetation patterns or soil types.\n"
    f"   - Note large-scale geographical features such as basins, mountains, coastal traits, or high-altitude snow, and relate them to regions within China.\n\n"

    f"2. **Linguistic and Textual Clues:**\n"
    f"   - Identify main scripts (Simplified Chinese), regional scripts (e.g., Traditional Chinese, Tibetan, Uyghur), or foreign languages.\n"
    f"   - Analyze any visible place names, administrative references, or local business terms suggesting regional origin.\n\n"

    f"3. **Architectural and Infrastructure Signatures:**\n"
    f"   - Characterize buildings (urban/rural, Northern/Southern/Ethnic architecture), roofing, balconies, etc.\n"
    f"   - Evaluate transportation infrastructure: road labels (G/S/X), signage forms, barriers, license plates, and public transport.\n"
    f"   - Spot regionally distinct features like streetlight designs or telephone area codes.\n\n"

    f"4. **Street View Provider and Meta Clues:**\n"
    f"   - Determine whether the imagery comes from Baidu, Google, or Tencent, and assess implications.\n"
    f"   - If parts of the street view car are visible, use its features to estimate location.\n"
    f"   - Mention any notable natural or human-made landmarks.\n\n"

    f"Use all of the above to explain why the real-world location corresponds precisely to coordinates "
    f"({latitude:.4f}, {longitude:.4f}).\n"
    f"Justify your conclusion using known geographic patterns in China (administrative, cultural, and physical), "
    f"geolocation strategies, and typical infrastructure distinctions across regions.\n\n"

    f"Structure your output in clear steps and aim for at least 200 words of detailed reasoning."
)

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

class GeoLingoDatasetBuilder:
    def __init__(self, image_base_dir: str, openai_api_key: str = None, base_url: str = None, raw_data_source: List[Dict[str, Any]] = None):
        self.image_base_dir = image_base_dir
        os.makedirs(self.image_base_dir, exist_ok=True)
        self.openai_client = None
        if openai and openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key, base_url=base_url)
                print("OpenAI client initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
        elif not openai:
             print("OpenAI library not available. CoT generation will be skipped or use simulation if implemented.")
        else:
            print("OpenAI API key not provided. CoT generation with GPT-4o will be skipped.")

        if raw_data_source is None:
            self.raw_data = [
                {"image_filename": "eiffel_tower_paris.jpg", "address": "Champ de Mars, 5 Av. Anatole France, 75007 Paris, France", "latitude": 48.8584, "longitude": 2.2945, "country": "France"},
                {"image_filename": "statue_of_liberty_ny.png", "address": "New York, NY 10004, USA", "latitude": 40.6892, "longitude": -74.0445, "country": "USA"},
                # For demonstration, one entry that will be a .txt dummy
                {"image_filename": "colosseum_rome.txt", "address": "Piazza del Colosseo, 1, 00184 Roma RM, Italy", "latitude": 41.8902, "longitude": 12.4922, "country": "Italy"},
                {"image_filename": "generic_suburbia.jpg", "address": "123 Main Street, Anytown, CA 90210, USA", "latitude": 34.0522, "longitude": -118.2437, "country": "USA"}
            ]
        else:
            self.raw_data = raw_data_source


    def build_dataset(self, num_items_to_build: int = -1, inspection_ratio: float = 0.01) -> List[GeoLingoDataItem]:
        dataset: List[GeoLingoDataItem] = []
        items_to_process = self.raw_data
        if 0 < num_items_to_build < len(self.raw_data):
            items_to_process = self.raw_data[:num_items_to_build]
        
        print(f"\nBuilding dataset with {len(items_to_process)} items...")
        for i, raw_item in enumerate(items_to_process):
            if raw_item["latitude"] == None or raw_item["longitude"] == None:
                continue
            image_path = os.path.join(self.image_base_dir, raw_item["image_filename"])
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}. Skipping this item.")
                continue

            cot_description, raw_response = None, None
            if self.openai_client:
                cot_description, raw_response = generate_cot_with_gpt4o(
                    self.openai_client, image_path, 
                    raw_item["latitude"], raw_item["longitude"]
                )
            else:
                cot_description = "[SKIPPED GPT-4o CALL - OpenAI client not available or API key missing] Simulated CoT for: " 
                print(f"  Skipped actual CoT generation for {raw_item['image_filename']}. Using placeholder.")
            
            if cot_description is None: # API call failed or was skipped and no fallback
                print(f"  Failed to generate CoT for {raw_item['image_filename']}. Skipping item.")
                continue

            dataset_item = GeoLingoDataItem(
                image_path=image_path,  latitude=raw_item["latitude"],
                longitude=raw_item["longitude"], 
                cot_description=cot_description, raw_model_response=raw_response
            )
            dataset.append(dataset_item)
            print(f"  Processed item {i+1}/{len(items_to_process)}: {raw_item['image_filename']}")

        if dataset and inspection_ratio > 0:
            num_to_inspect = max(1, int(len(dataset) * inspection_ratio))
            print(f"\nSimulating Quality Assurance: Review {num_to_inspect} generated CoT descriptions.")
            # In a real pipeline, this involves manual review.
        
        print(f"\nDataset building complete. Total items: {len(dataset)}")
        return dataset

    def save_dataset_to_json(self, dataset: List[GeoLingoDataItem], output_filepath: str):
        try:
            dataset_as_dicts = [asdict(item) for item in dataset]
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset_as_dicts, f, indent=4, ensure_ascii=False)
            print(f"\nDataset successfully saved to: {output_filepath}")
        except Exception as e:
            print(f"Error saving dataset to {output_filepath}: {e}")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    print(OPENAI_API_KEY, BASE_URL)

    if not OPENAI_API_KEY and openai:
        print("Warning: OPENAI_API_KEY environment variable not set. GPT-4o calls will be skipped.")
        print("CoT descriptions will be placeholders.")
    elif not openai:
        print("OpenAI library not installed. Cannot make GPT-4o calls.")

    IMAGE_BASE_DIRECTORY = "./dataset/tuxun_streetview_images_02"
    OUTPUT_JSON_FILE = "./geolingo_finetuning_dataset.json"

    '''
    self.raw_data = [
                {"image_filename": "eiffel_tower_paris.jpg", "address": "Champ de Mars, 5 Av. Anatole France, 75007 Paris, France", "latitude": 48.8584, "longitude": 2.2945, "country": "France"},
                {"image_filename": "statue_of_liberty_ny.png", "address": "New York, NY 10004, USA", "latitude": 40.6892, "longitude": -74.0445, "country": "USA"},
                # For demonstration, one entry that will be a .txt dummy
                {"image_filename": "colosseum_rome.txt", "address": "Piazza del Colosseo, 1, 00184 Roma RM, Italy", "latitude": 41.8902, "longitude": 12.4922, "country": "Italy"},
                {"image_filename": "generic_suburbia.jpg", "address": "123 Main Street, Anytown, CA 90210, USA", "latitude": 34.0522, "longitude": -118.2437, "country": "USA"}
            ]
    '''
    # Load raw_data from external JSON file if it exists
    RAW_DATA_JSON_PATH = os.path.join(IMAGE_BASE_DIRECTORY, "tuxun_data_collected_api_final_v3.json")
    raw_data_source = None
    if os.path.exists(RAW_DATA_JSON_PATH):
        try:
            with open(RAW_DATA_JSON_PATH, "r", encoding="utf-8") as f:
                raw_data_source = json.load(f)
            print(f"Loaded raw_data from {RAW_DATA_JSON_PATH} ({len(raw_data_source)} items).")
        except Exception as e:
            print(f"Error loading raw_data from {RAW_DATA_JSON_PATH}: {e}")
    else:
        print(f"Warning: {RAW_DATA_JSON_PATH} not found. Using default raw_data.")

    for data in raw_data_source:
        data['image_filename']="round_"+str(data['round_number'])+"_pano_"+data['current_street_view_pano_id']+"_face_0_front_corrected_z5.jpg"
        data['latitude']=data['correct_latitude']
        data['longitude']=data['correct_longitude']

    print(f"Initializing dataset builder. Image files will be referenced from/created in: '{IMAGE_BASE_DIRECTORY}'")
    builder = GeoLingoDatasetBuilder(image_base_dir=IMAGE_BASE_DIRECTORY, openai_api_key=OPENAI_API_KEY, base_url=BASE_URL, raw_data_source=raw_data_source)

    # Limit number of items for testing to avoid excessive API calls/costs initially
    # Set to -1 to process all items in raw_data
    NUM_ITEMS_TO_PROCESS_FOR_TESTING = -1 
    print(f"\nBuilding dataset (processing up to {NUM_ITEMS_TO_PROCESS_FOR_TESTING} items for this run)...")
    geolingo_dataset = builder.build_dataset(num_items_to_build=NUM_ITEMS_TO_PROCESS_FOR_TESTING, inspection_ratio=0.5)

    if geolingo_dataset:
        builder.save_dataset_to_json(geolingo_dataset, OUTPUT_JSON_FILE)
    else:
        print("No dataset was built, skipping save.")

    print("\n--- Script Finished ---")
    if OPENAI_API_KEY and openai:
        print("Remember to check your OpenAI account for API usage costs.") 