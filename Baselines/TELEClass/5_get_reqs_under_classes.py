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

def transform_json(input_file: str, output_file: str):
    input_data = read_json(input_file)
    
    new_data: List[Dict] = []
    class_to_reqs: Dict[str, List[str]] = {}  
    for item in input_data:
        req = item['req']
        llm_core_classes = item['LLM_core_classes']
        
        for class_name in llm_core_classes:
            if class_name in class_to_reqs:
                if req not in class_to_reqs[class_name]:
                    class_to_reqs[class_name].append(req)
            else:
                class_to_reqs[class_name] = [req]
    
    for class_name, req_list in class_to_reqs.items():
        new_item = {
            "class": class_name,
            "req": req_list
        }
        new_data.append(new_item)
    
    save_json(new_data, output_file)

if __name__ == "__main__":
    input_file_path = r"path/to/3_classes_from_core_classes.json"  
    output_file_path = r"path/to/reqs_under_classes.json"  
    transform_json(input_file_path, output_file_path)