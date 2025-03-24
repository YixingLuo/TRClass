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
        print(f"save {file_path} error: {e}")

def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def extract_labels(df, labels):
    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def parse_hierarchy(hierarchy: Dict) -> tuple[Set[str], Set[str]]:
    guzhangzhenduan_classes=set({"故障诊断"})
    shujuyouxiaoxingpanduan_classes=set({"数据有效性判断"})
    return guzhangzhenduan_classes,shujuyouxiaoxingpanduan_classes
def map_to_level_4_df(df, reqs: List[str], guzhangzhenduan_classes: Set[str],shujuyouxiaoxingpanduan_classes: Set[str]) -> pd.DataFrame:
    result = pd.DataFrame({'req': reqs})
    level_4_data = []
    for i in range(df.shape[0]):
        req = reqs[i]
        guzhangzhenduan=shujuyouxiaoxingpanduan=0
        labels = df.iloc[i, 1:].to_numpy()  
        for j, label in enumerate(LABEL):
            if labels[j] == 1:
                if label in guzhangzhenduan_classes:
                    guzhangzhenduan = 1
                if label in shujuyouxiaoxingpanduan_classes:
                    shujuyouxiaoxingpanduan = 1
        level_4_data.append({"故障诊断":guzhangzhenduan,"数据有效性判断":shujuyouxiaoxingpanduan})
    result = pd.concat([result, pd.DataFrame(level_4_data)], axis=1)
    return result

def map_to_level_4_json(pred_json: List[Dict], guzhangzhenduan_classes: Set[str],shujuyouxiaoxingpanduan_classes: Set[str]) -> List[Dict]:
    result = []
    for item in pred_json[:-1]:
        req = item['req']
        pred_classes = item['labels']  
        level_4_classes = set()  
        for label in pred_classes:
            if label in guzhangzhenduan_classes:
                level_4_classes.add('故障诊断')
            if label in shujuyouxiaoxingpanduan_classes:
                level_4_classes.add('数据有效性判断')
        level_4 = {'req': req, 'class': list(level_4_classes)}  
        result.append(level_4)
    return result

def process_level_4(true_path, pred_path, pred_json_path, hierarchy_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    pred_json = read_json(pred_json_path)
    hierarchy = read_json(hierarchy_path)
    

    assert len(true_df) == len(pred_df) == len(pred_json)-1
    reqs = true_df['req'].tolist()  

    guzhangzhenduan_classes,shujuyouxiaoxingpanduan_classes = parse_hierarchy(hierarchy)
    
    true_level_4_df = map_to_level_4_df(true_df, reqs, guzhangzhenduan_classes,shujuyouxiaoxingpanduan_classes)
    pred_level_4_df = map_to_level_4_df(pred_df, reqs, guzhangzhenduan_classes,shujuyouxiaoxingpanduan_classes)
    
    pred_json_level_4 = map_to_level_4_json(pred_json, guzhangzhenduan_classes,shujuyouxiaoxingpanduan_classes)
    
    true_level_4_df.to_excel(os.path.join(output_dir, 'true_level_4.xlsx'), index=False)
    pred_level_4_df.to_excel(os.path.join(output_dir, 'pred_level_4.xlsx'), index=False)
    save_json(pred_json_level_4, os.path.join(output_dir, 'pred_json_level_4.json'))
if __name__ == "__main__":
    true_path = r"path/to/true_labels,xlsx"
    pred_path = r"path/to/flatten.xlsx"
    pred_json_path = r"path/to/flatten.json"
    hierarchy_path = r"path/to/hierarchy.json"
    output_dir = r"path/to/level_4"
    process_level_4(true_path, pred_path, pred_json_path, hierarchy_path, output_dir)