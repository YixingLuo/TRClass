import pandas as pd
import json

def create_tree_from_excel(excel_path: str, output_json_path: str) -> None:
    df = pd.read_excel(excel_path)
    data = df.values.tolist()
    
    root = {"name": "root", "children": []}
    level1_nodes = {}
    
    for row in data:
        req_desc = row[0]  
        levels = [x for x in row[1:] if pd.notna(x) and x != ""]  
        
        if not levels:  
            continue
            
        level1_name = levels[0]
        if level1_name not in level1_nodes:
            level1_nodes[level1_name] = {
                "name": level1_name,
                "children": []
            }
            root["children"].append(level1_nodes[level1_name])
            
        current_node = level1_nodes[level1_name]
        
        for level_name in levels[1:]:
            found = False
            for child in current_node["children"]:
                if child["name"] == level_name:
                    current_node = child
                    found = True
                    break
                    
            if not found:
                new_node = {
                    "name": level_name,
                    "children": []
                }
                current_node["children"].append(new_node)
                current_node = new_node
                
        if "leafnodes" not in current_node:
            current_node["leafnodes"] = []
        current_node["leafnodes"].append(req_desc)
    
    def ensure_leafnodes(node):
        if "children" in node:
            if not node["children"] and "leafnodes" not in node:
                node["leafnodes"] = []
            for child in node["children"]:
                ensure_leafnodes(child)
    
    ensure_leafnodes(root)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    excel_file = r"path/to/true_lables.xlsx"
    output_file = r"path/to/hierarchy.json"
    
    create_tree_from_excel(excel_file, output_file)