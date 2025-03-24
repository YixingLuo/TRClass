import json
from typing import List, Dict

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

def match_reqs_to_classes(req_embeddings_file: str, terms_file: str, output_file: str):
    req_embeddings_data = read_json(req_embeddings_file)
    terms_data = read_json(terms_file)
    
    req_to_embedding = {item['req']: item['embedding'] for item in req_embeddings_data}
    
    class_to_terms = {item['class']: item['terms'] for item in terms_data}
    
    result_data = []
    i=1
    for class_name, terms in class_to_terms.items():
        print(f"第{i}个class:{class_name}")
        terms_lower = [term.lower() for term in terms]
        
        matched_reqs = []
        matched_embeddings = []
        for req, embedding in req_to_embedding.items():
            req_lower = req.lower()
            if any(term in req_lower for term in terms_lower):
                matched_reqs.append(req)
                matched_embeddings.append(embedding)
        
        if matched_reqs:
            new_item = {
                "class": class_name,
                "reqs": matched_reqs,
                "req_embeddings": matched_embeddings
            }
            result_data.append(new_item)
        print(f"{len(matched_reqs)} reqs")
    save_json(result_data, output_file)

if __name__ == "__main__":
    req_embeddings_file_path = r"path/to/req_embeddings.json"  
    terms_file_path = r"path/to/combined_terms.json"      
    output_file_path = r"path/to/req_embeddings_for_classes.json"         
    match_reqs_to_classes(req_embeddings_file_path, terms_file_path, output_file_path)