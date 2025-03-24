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

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def assign_classes_to_reqs(req_embeddings_file: str, class_embeddings_file: str, output_file: str):
    req_data = read_json(req_embeddings_file)
    class_data = read_json(class_embeddings_file)
    
    class_to_embedding = {item['class']: np.array(item['embedding']) for item in class_data}
    
    new_data = []
    
    for req_item in req_data:
        req = req_item['req']
        req_embedding = np.array(req_item['embedding'])
        
        similarities = {}
        for class_name, class_embedding in class_to_embedding.items():
            similarity = cosine_similarity(req_embedding, class_embedding)
            similarities[class_name] = similarity
        
        sorted_classes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        class_names = [item[0] for item in sorted_classes]
        similarity_scores = [item[1] for item in sorted_classes]
        
        diff_scores = [similarity_scores[i] - similarity_scores[i + 1] for i in range(len(similarity_scores) - 1)]
        
        if diff_scores:
            max_diff_index = np.argmax(diff_scores)
            final_classes = class_names[:max_diff_index + 1]  
        else:
            final_classes = class_names  
        
        new_item = {
            "req": req,
            "refined_core_classes": final_classes
        }
        new_data.append(new_item)
    
    save_json(new_data, output_file)

if __name__ == "__main__":
    req_embeddings_file_path = r"path/to/req_embeddings.json"
    class_embeddings_file_path = r"path/to/class_embeddings.json"
    output_file_path = r"path/to/refined_core_classes.json"
    assign_classes_to_reqs(req_embeddings_file_path, class_embeddings_file_path, output_file_path)