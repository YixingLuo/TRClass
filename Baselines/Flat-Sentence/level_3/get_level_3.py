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

def parse_hierarchy(hierarchy: Dict) -> tuple[Set[str], Set[str],Set[str], Set[str],Set[str], Set[str],Set[str], Set[str]]:
    zitaikongzhi_classes=set({"姿态控制"})
    zitaiqueding_classes=set({"姿态确定"})
    yunxingshibaozhang_classes=set({"故障诊断","数据有效性判断"})
    moshiguanli_classes = set({"模式管理"})
    guidaojisuan_classes=set({"轨道计算"})
    jingduyueshu_classes=set({"精度约束"})
    kongjianyueshu_classes = set({"空间约束"})
    shijianyueshu_classes=set({"时间约束"})
    return zitaikongzhi_classes,zitaiqueding_classes,yunxingshibaozhang_classes,moshiguanli_classes,guidaojisuan_classes,jingduyueshu_classes,kongjianyueshu_classes,shijianyueshu_classes
def map_to_level_3_df(df, reqs: List[str], zitaikongzhi_classes: Set[str],zitaiqueding_classes: Set[str],yunxingshibaozhang_classes: Set[str],moshiguanli_classes: Set[str],guidaojisuan_classes: Set[str],jingduyueshu_classes: Set[str],kongjianyueshu_classes: Set[str],shijianyueshu_classes: Set[str]) -> pd.DataFrame:
    result = pd.DataFrame({'req': reqs})
    level_3_data = []
    for i in range(df.shape[0]):
        req = reqs[i]
        zitaikongzhi=zitaiqueding=yunxingshibaozhang=moshiguanli=guidaojisuan=jingduyueshu=kongjianyueshu=shijianyueshu=0
        labels = df.iloc[i, 1:].to_numpy()  
        for j, label in enumerate(LABEL):
            if labels[j] == 1:
                if label in zitaikongzhi_classes:
                    zitaikongzhi = 1
                if label in zitaiqueding_classes:
                    zitaiqueding = 1
                if label in yunxingshibaozhang_classes:
                    yunxingshibaozhang = 1
                if label in moshiguanli_classes:
                    moshiguanli = 1
                if label in guidaojisuan_classes:
                    guidaojisuan = 1
                if label in jingduyueshu_classes:
                    jingduyueshu = 1
                if label in kongjianyueshu_classes:
                    kongjianyueshu = 1
                if label in shijianyueshu_classes:
                    shijianyueshu = 1
        level_3_data.append({"姿态控制":zitaikongzhi,"姿态确定":zitaiqueding, "运行时保障":yunxingshibaozhang,"模式管理":moshiguanli,"轨道计算":guidaojisuan, "精度约束":jingduyueshu,"空间约束":kongjianyueshu,"时间约束":shijianyueshu})
    result = pd.concat([result, pd.DataFrame(level_3_data)], axis=1)
    return result

def map_to_level_3_json(pred_json: List[Dict], zitaikongzhi_classes: Set[str],zitaiqueding_classes: Set[str],yunxingshibaozhang_classes: Set[str],moshiguanli_classes: Set[str],guidaojisuan_classes: Set[str],jingduyueshu_classes: Set[str],kongjianyueshu_classes: Set[str],shijianyueshu_classes: Set[str]) -> List[Dict]:
    result = []
    for item in pred_json[:-1]:
        req = item['req']
        pred_classes = item['labels']  
        level_3_classes = set()  
        for label in pred_classes:
            if label in zitaikongzhi_classes:
                level_3_classes.add('姿态控制')
            if label in zitaiqueding_classes:
                level_3_classes.add('姿态确定')
            if label in yunxingshibaozhang_classes:
                level_3_classes.add('运行时保障')
            if label in moshiguanli_classes:
                level_3_classes.add('模式管理')
            if label in guidaojisuan_classes:
                level_3_classes.add('轨道计算')
            if label in jingduyueshu_classes:
                level_3_classes.add('精度约束')
            if label in kongjianyueshu_classes:
                level_3_classes.add('空间约束')
            if label in shijianyueshu_classes:
                level_3_classes.add('时间约束')
        level_3 = {'req': req, 'class': list(level_3_classes)}  
        result.append(level_3)
    return result

def process_level_3(true_path, pred_path, pred_json_path, hierarchy_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    pred_json = read_json(pred_json_path)
    hierarchy = read_json(hierarchy_path)
    
    assert len(true_df) == len(pred_df) == len(pred_json)-1
    reqs = true_df['req'].tolist()  
    
    zitaikongzhi_classes,zitaiqueding_classes,yunxingshibaozhang_classes,moshiguanli_classes,guidaojisuan_classes,jingduyueshu_classes,kongjianyueshu_classes,shijianyueshu_classes = parse_hierarchy(hierarchy)
    
    true_level_3_df = map_to_level_3_df(true_df, reqs, zitaikongzhi_classes,zitaiqueding_classes,yunxingshibaozhang_classes,moshiguanli_classes,guidaojisuan_classes,jingduyueshu_classes,kongjianyueshu_classes,shijianyueshu_classes)
    pred_level_3_df = map_to_level_3_df(pred_df, reqs, zitaikongzhi_classes,zitaiqueding_classes,yunxingshibaozhang_classes,moshiguanli_classes,guidaojisuan_classes,jingduyueshu_classes,kongjianyueshu_classes,shijianyueshu_classes)
    
    pred_json_level_3 = map_to_level_3_json(pred_json, zitaikongzhi_classes,zitaiqueding_classes,yunxingshibaozhang_classes,moshiguanli_classes,guidaojisuan_classes,jingduyueshu_classes,kongjianyueshu_classes,shijianyueshu_classes)
    
    true_level_3_df.to_excel(os.path.join(output_dir, 'true_level_3.xlsx'), index=False)
    pred_level_3_df.to_excel(os.path.join(output_dir, 'pred_level_3.xlsx'), index=False)
    save_json(pred_json_level_3, os.path.join(output_dir, 'pred_json_level_3.json'))

if __name__ == "__main__":
    true_path = r"path/to/true_labels,xlsx"
    pred_path = r"path/to/flatten.xlsx"
    pred_json_path = r"path/to/flatten.json"
    hierarchy_path = r"path/to/hierarchy.json"
    output_dir = r"path/to/level_3"
    process_level_3(true_path, pred_path, pred_json_path, hierarchy_path, output_dir)