import pandas as pd
import json
from openai import OpenAI

def read_excel_and_build_paths(input_excel):
    df = pd.read_excel(input_excel)

    items = []
    for _, row in df.iterrows():
        levels = [row['Level1_1'], row['Level2_1'], row['Level3_1'], row['Level4_1']]
        path = '—'.join(filter(pd.notna, levels))  
        siblings = []

        level_index = len([lv for lv in levels if pd.notna(lv)]) - 1
        sibling_column = f"Level{level_index+1}_1"

        if sibling_column in df.columns:
            siblings = list(df[df[sibling_column].notna()][sibling_column].unique())
            siblings = [s for s in siblings if s != levels[level_index]]  
        items.append({
            "path": path,
            "siblings": siblings
        })

    return items


def ask_llm(path, siblings):
    name = path.split('—')[-1]
    parent = '—'.join(path.split('—')[1:-1])
    siblings_str = ','.join(siblings)

    model_input = f'''
    ['{name}'] is a requirement class in aerospace control software requirements classification tree and is the subclass of ['{parent}']. 
    Please generate 10 additional key terms about the ['{name}'] that are relevant to ['{name}'] but irrelevant to ['{siblings_str}']. 
    Please split the additional key terms using commas. Please adhere to the following guidelines:
    1. The maximum number of additional key terms is 10.
    2. Provide your answer only in JSON format as follows: {{"additional key terms":["additional key term 1","additional key term 2",...]}}.
    3. Ensure no additional text or explanations are included outside this JSON format.
    '''

    client = OpenAI(
        api_key="",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        messages=[
            {"role": "system", "content": "You are an expert in feature extraction for aerospace control software requirements."},
            {"role": "user", "content": model_input}
        ]
    )

    response_content = completion.choices[0].message.content
    if response_content.startswith("```json"):
        response_content = response_content.strip("```json").strip("```").strip()

    if "</think>" in response_content:
        json_start = response_content.find('{')
        json_end = response_content.rfind('}') + 1
        json_str = response_content[json_start:json_end]
        return json.loads(json_str)
    else:
        return json.loads(response_content)


def build_tree_node(name, children_names, path, siblings):
    feature = ask_llm(path, siblings).get("additional key terms", [])
    return {
        "name": name,
        "children": [],
        "feature": feature
    }


def build_tree_hierarchy(items, parent_path=""):
    tree = []

    children_map = {}
    for item in items:
        path_parts = item["path"].split('—')
        parent = '—'.join(path_parts[:-1])
        name = path_parts[-1]

        if parent == parent_path:
            if parent_path not in children_map:
                children_map[parent_path] = []
            children_map[parent_path].append(item)

    for item in children_map.get(parent_path, []):
        name = item["path"].split('—')[-1]
        path = item["path"]
        siblings = item["siblings"]

        node = build_tree_node(name, [], path, siblings)
        node["children"] = build_tree_hierarchy(items, parent_path=path)
        tree.append(node)

    return tree


def process_excel_to_tree(input_excel, output_file, log_file):
    items = read_excel_and_build_paths(input_excel)

    tree_structure = build_tree_hierarchy(items)

    with open(output_file, "w", encoding="utf-8") as f:
        print(f"write file: {output_file}")
        print(f"tree_structure: {json.dumps(tree_structure, ensure_ascii=False, indent=4)}")  
        json.dump(tree_structure, f, ensure_ascii=False, indent=4)
        print(f"done: {output_file}")

    with open(log_file, "w", encoding="utf-8") as log_f:
        for node in tree_structure:
            log_features_recursive(node, log_f)


def log_features_recursive(node, log_f):
    log_f.write(f"Path: {node['name']}\n")
    log_f.write(f"Feature: {json.dumps(node['feature'], ensure_ascii=False)}\n\n")
    for child in node["children"]:
        log_features_recursive(child, log_f)


input_excel = 'path/to/true_labels.xlsx'
output_file = 'path/to/llm_10_terms.json'
log_file = 'path/to/log.log'

process_excel_to_tree(input_excel, output_file, log_file)