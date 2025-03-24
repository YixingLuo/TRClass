import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List, Dict

def read_excel(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return pd.DataFrame()

def save_json(data: List[Dict], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"saved to {file_path}")
    except Exception as e:
        print(f"save {file_path} error: {e}")

def generate_embeddings(excel_file: str, output_file: str, local_model_path: str):
    print(f"loading model: {local_model_path}")
    model = SentenceTransformer(local_model_path, device='cuda')  
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("CPU")
    
    df = read_excel(excel_file)
    
    req_descriptions = df['Requirements Description'].astype(str).tolist()
    
    new_data = []
    
    for index, req in enumerate(req_descriptions, start=1):
        embedding = model.encode([req], convert_to_numpy=True)[0]  
        embedding_list = embedding.tolist()
        new_item = {
            "req": req,
            "embedding": embedding_list
        }
        new_data.append(new_item)
        save_json(new_data, output_file)

if __name__ == "__main__":
    excel_file_path = r"path/to/true_labels.xlsx"  
    output_file_path = r"path/to/req_embeddings.json"  
    local_model_path = r"path/to/all-mpnet-base-v2"  
    generate_embeddings(excel_file_path, output_file_path, local_model_path)