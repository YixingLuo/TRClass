import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_leaf_nodes(node, result):
    if not node.get('children'):  
        result.append({
            "class": node["name"],
            "terms": node["feature"] if node["feature"] is not None else []
        })
    else:
        for child in node["children"]:
            extract_leaf_nodes(child, result)

def save_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_json(input_path, output_path):
    data = read_json(input_path)
    
    result = []
    extract_leaf_nodes(data, result)
    
    save_json(result, output_path)
    print(f"saved to {output_path}")

if __name__ == "__main__":
    input_path = "path/to/llm_10_terms.json"  
    output_path = "path/to/llm_10_terms.json"  
    process_json(input_path, output_path)