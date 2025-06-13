import json
import os

def load_game_data_from_json(filepath: str) -> list[dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Successfully loaded {len(data)} records from '{filepath}'.")
                return data
            else:
                print(f"Warning: The JSON structure in '{filepath}' is not a list.")
                return []
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON in file '{filepath}': {e}")
        return []
    except Exception as e:
        print(f"Error: An exception occurred while reading '{filepath}': {e}")
        return []

def save_data_to_json(data_to_save: list[dict], output_filepath: str):
    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to: {output_filepath}")
    except Exception as e:
        print(f"Error: Failed to save data: {e}")

def is_valid_coord(x):
    try:
        val = float(x)
        return not (val is None or val != val) 
    except:
        return False

base_root = "/root/autodl-tmp/dataset"
folder_names = [f"tuxun_streetview_images_0{i}" for i in range(2, 8)]
all_processed_records = []

for folder in folder_names:
    base_dir = os.path.join(base_root, folder)
    json_file = os.path.join(base_dir, "tuxun_data_collected_api_final_v2.json")
    game_records = load_game_data_from_json(json_file)

    if not game_records:
        continue

    for record in game_records:
        lat = record.get("correct_latitude")
        lon = record.get("correct_longitude")
        if not (is_valid_coord(lat) and is_valid_coord(lon)):
            continue
        pano_id = record.get("current_street_view_pano_id")
        round_num = record.get("round_number")
        for face in ["front","right","left","back"]:
            filename = f"round_{round_num}_pano_{pano_id}_face_{['front','right','back','left'].index(face)}_{face}_corrected_z5.jpg"
            full_path = os.path.join(base_dir, filename)

            processed_record = {
                "address": "None",
                "country": "China",
                "longitude": float(lon),
                "latitude": float(lat),
                "image_filename": full_path
            }
            all_processed_records.append(processed_record)

print(f"\nTotal processed image samples: {len(all_processed_records)}")

output_filename = os.path.join(base_root, "merged_game_locations_val.json")
save_data_to_json(all_processed_records, output_filename)