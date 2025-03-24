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

def merge_terms_json(file1_path: str, file2_path: str, output_file: str):
    data1 = read_json(file1_path)  
    data2 = read_json(file2_path)  
    
    terms_dict1 = {item['class']: item['terms'] for item in data1}
    
    terms_dict2 = {item['class']: item['terms'] for item in data2}
    
    new_data = []
    for class_name in terms_dict1.keys():
        if class_name in terms_dict2:
            combined_terms = terms_dict1[class_name][:10] + terms_dict2[class_name][:10]
            new_item = {
                "class": class_name,
                "terms": combined_terms
            }
            new_data.append(new_item)
        else:
            print(f"warning：{class_name} is not in the second file,step over!")
    
    for class_name in terms_dict2.keys():
        if class_name not in terms_dict1:
            print(f"warning：{class_name} is not in the first file,step over!")
    
    save_json(new_data, output_file)

# 使用示例
if __name__ == "__main__":
    file1_path = r"path/to/llm_10_terms.json"  
    file2_path = r"path/to/another_10_terms_with_corpus.json"          
    output_file_path = r"path/to/combined_terms.json"        
    merge_terms_json(file1_path, file2_path, output_file_path)