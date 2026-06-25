import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

METADATA_FILE = "metadata.json"

def save_metadata(path: str, data: dict):
    os.makedirs(path, exist_ok=True)
    data["time"] = str(datetime.now(ZoneInfo("America/New_York")))
    file_path = Path(f"{path}/{METADATA_FILE}")
    assert not file_path.exists(), "Conflicting with existing metadata."
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def add_metadata(path: str, key: str, value):
    file_path = Path(f"{path}/{METADATA_FILE}")
    assert os.path.exists(file_path)
    
    with open(file_path, 'r') as file: 
        data = json.load(file)
    
    data[key] = value
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_metadata(dir_path: str):
    with open(f"{dir_path}/{METADATA_FILE}", 'r') as file:
        return json.load(file)
