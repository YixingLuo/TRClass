import json
import pandas as pd
import os
from typing import List, Dict, Set

LABEL = ["初始化", "可复用需求", "可维护需求", "可靠性需求", "姿态控制", "姿态确定", "安全性需求", "控制输出", "故障诊断", "数据有效性判断", "数据采集", "时间约束", "模式管理", "空间约束", "精度约束", "轨道计算", "遥控", "遥测"]

def read_json(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return {}

def save_json(data: List[Dict], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"saved to {file_path}")
    except Exception as e:
        print(f"saved {file_path} error: {e}")

def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def extract_labels(df, labels):
    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def parse_hierarchy(hierarchy: Dict) -> tuple[Set[str], Set[str]]:
    functional_classes = set({"初始化", "数据采集", "控制输出", "姿态控制", "姿态确定", "故障诊断", "数据有效性判断", "模式管理", "轨道计算", "遥控", "遥测"})
    nonfunctional_classes = set({"精度约束", "空间约束", "时间约束", "安全性需求", "可靠性需求", "可复用需求", "可维护需求"})
    return functional_classes, nonfunctional_classes

def map_to_level_1_df(df, reqs: List[str], functional_classes: Set[str], nonfunctional_classes: Set[str]) -> pd.DataFrame:
    result = pd.DataFrame({'req': reqs})
    level_1_data = []
    for i in range(df.shape[0]):
        req = reqs[i]
        func = 0
        nonfunc = 0
        labels = df.iloc[i, 1:].to_numpy()  
        for j, label in enumerate(LABEL):
            if labels[j] == 1:
                if label in functional_classes:
                    func = 1
                if label in nonfunctional_classes:
                    nonfunc = 1
        level_1_data.append({'功能需求': func, '非功能需求': nonfunc})
    result = pd.concat([result, pd.DataFrame(level_1_data)], axis=1)
    return result

def map_to_level_1_json(pred_json: List[Dict], functional_classes: Set[str], nonfunctional_classes: Set[str]) -> List[Dict]:
    result = []
    for item in pred_json[:-1]:
        req = item['req']
        pred_classes = item['labels']  
        level_1_classes = set()  
        for label in pred_classes:
            if label in functional_classes:
                level_1_classes.add('功能需求')
            if label in nonfunctional_classes:
                level_1_classes.add('非功能需求')
        level_1 = {'req': req, 'class': list(level_1_classes)}  
        result.append(level_1)
    return result


def process_level_1(true_path, pred_path, pred_json_path, hierarchy_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    pred_json = read_json(pred_json_path)
    hierarchy = read_json(hierarchy_path)
    
    assert len(true_df) == len(pred_df) == len(pred_json)-1
    
    reqs = true_df['req'].tolist()  
    
    functional_classes, nonfunctional_classes = parse_hierarchy(hierarchy)
    
    true_level_1_df = map_to_level_1_df(true_df, reqs, functional_classes, nonfunctional_classes)
    pred_level_1_df = map_to_level_1_df(pred_df, reqs, functional_classes, nonfunctional_classes)
    
    pred_json_level_1 = map_to_level_1_json(pred_json, functional_classes, nonfunctional_classes)
    
    true_level_1_df.to_excel(os.path.join(output_dir, 'path/to/true_level_1.xlsx'), index=False)
    pred_level_1_df.to_excel(os.path.join(output_dir, 'path/to/pred_level_1.xlsx'), index=False)
    save_json(pred_json_level_1, os.path.join(output_dir, 'path/to/pred_json_level_1.json'))

# 使用示例
if __name__ == "__main__":
    true_path = r"path/to/true_labels.xlsx"
    pred_path = r"path/to/hier.xlsx"
    pred_json_path = r"path/to/hier.json"
    hierarchy_path = r"path/to/hierarchy.json"
    output_dir = r"path/to/level_1"
    process_level_1(true_path, pred_path, pred_json_path, hierarchy_path, output_dir)