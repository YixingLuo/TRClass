import json
from typing import List, Dict
import numpy as np

def read_json(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return []

def save_json(data: List[Dict], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"saved to {file_path}")
    except Exception as e:
        print(f"save {file_path} error: {e}")

def compute_average_embeddings(input_file: str, output_file: str):
    input_data = read_json(input_file)
    
    new_data = []
    
    for item in input_data:
        class_name = item['class']
        req_embeddings = item['req_embeddings']
        
        if not req_embeddings:
            print(f"warning: req_embeddings of {class_name} is None,step over!")
            continue
        
        embeddings = np.array(req_embeddings)
        
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        new_item = {
            "class": class_name,
            "embedding": avg_embedding
        }
        new_data.append(new_item)
    
    save_json(new_data, output_file)

if __name__ == "__main__":
    input_file_path = r"path/to/req_embeddings_for_classes.json"
    output_file_path = r"path/to/class_embeddings.json"
    compute_average_embeddings(input_file_path, output_file_path)