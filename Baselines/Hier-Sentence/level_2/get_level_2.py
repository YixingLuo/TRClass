import json
import pandas as pd
import os
from typing import List, Dict, Set
import numpy as np

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

def parse_hierarchy(hierarchy: Dict) -> tuple[Set[str], Set[str],Set[str], Set[str],Set[str], Set[str],Set[str], Set[str],Set[str], Set[str],Set[str]]:
    chushihua_classes=set({"初始化"})
    shujucaiji_classes=set({"数据采集"})
    kongzhishuchu_classes=set({"控制输出"})
    kongzhijisuan_classes = set({"姿态控制", "姿态确定", "故障诊断", "数据有效性判断", "模式管理", "轨道计算"})
    yaokong_classes=set({"遥控"})
    yaoce_classes=set({"遥测"})
    xingnengxuqiu_classes = set({"精度约束", "空间约束", "时间约束"})
    kefuyongxuqiu_classes=set({"可复用需求"})
    keweihuxuqiu_classes=set({"可维护需求"})
    kekaoxingxuqiu_classes=set({"可靠性需求"})
    anquanxingxuqiu_classes=set({"安全性需求"})
    return chushihua_classes,shujucaiji_classes,kongzhishuchu_classes,kongzhijisuan_classes,yaokong_classes,yaoce_classes,xingnengxuqiu_classes,kefuyongxuqiu_classes,keweihuxuqiu_classes,kekaoxingxuqiu_classes,anquanxingxuqiu_classes

def map_to_level_2_df(df, reqs: List[str], chushihua_classes: Set[str],shujucaiji_classes: Set[str],kongzhishuchu_classes: Set[str],kongzhijisuan_classes: Set[str],yaokong_classes: Set[str],yaoce_classes: Set[str],xingnengxuqiu_classes: Set[str],kefuyongxuqiu_classes: Set[str],keweihuxuqiu_classes: Set[str],kekaoxingxuqiu_classes: Set[str],anquanxingxuqiu_classes: Set[str]) -> pd.DataFrame:
    result = pd.DataFrame({'req': reqs})
    level_2_data = []
    for i in range(df.shape[0]):
        req = reqs[i]
        chushihua=shujucaiji=kongzhishuchu=kongzhijisuan=yaokong=yaoce=xingnengxuqiu=kefuyongxuqiu=keweihuxuqiu=kekaoxingxuqiu=anquanxingxuqiu=0
        labels = df.iloc[i, 1:].to_numpy()  
        for j, label in enumerate(LABEL):
            if labels[j] == 1:
                if label in chushihua_classes:
                    chushihua = 1
                if label in shujucaiji_classes:
                    shujucaiji = 1
                if label in kongzhishuchu_classes:
                    kongzhishuchu = 1
                if label in kongzhijisuan_classes:
                    kongzhijisuan = 1
                if label in yaokong_classes:
                    yaokong = 1
                if label in yaoce_classes:
                    yaoce = 1
                if label in xingnengxuqiu_classes:
                    xingnengxuqiu = 1
                if label in kefuyongxuqiu_classes:
                    kefuyongxuqiu = 1
                if label in keweihuxuqiu_classes:
                    keweihuxuqiu = 1
                if label in kekaoxingxuqiu_classes:
                    kekaoxingxuqiu = 1
                if label in anquanxingxuqiu_classes:
                    anquanxingxuqiu = 1
        level_2_data.append({"初始化":chushihua,"数据采集":shujucaiji, "控制输出":kongzhishuchu,"控制计算":kongzhijisuan,"遥控":yaokong, "遥测":yaoce,"性能需求":xingnengxuqiu,"可复用需求":kefuyongxuqiu, "可维护需求":keweihuxuqiu, "可靠性需求":kekaoxingxuqiu, "安全性需求":anquanxingxuqiu})
    result = pd.concat([result, pd.DataFrame(level_2_data)], axis=1)
    return result

def map_to_level_2_json(pred_json: List[Dict], chushihua_classes: Set[str],shujucaiji_classes: Set[str],kongzhishuchu_classes: Set[str],kongzhijisuan_classes: Set[str],yaokong_classes: Set[str],yaoce_classes: Set[str],xingnengxuqiu_classes: Set[str],kefuyongxuqiu_classes: Set[str],keweihuxuqiu_classes: Set[str],kekaoxingxuqiu_classes: Set[str],anquanxingxuqiu_classes: Set[str]) -> List[Dict]:
    result = []
    for item in pred_json[:-1]:
        req = item['req']
        pred_classes = item['labels']  
        level_2_classes = set()  
        for label in pred_classes:
            if label in chushihua_classes:
                level_2_classes.add('初始化')
            if label in shujucaiji_classes:
                level_2_classes.add('数据采集')
            if label in kongzhishuchu_classes:
                level_2_classes.add('控制输出')
            if label in kongzhijisuan_classes:
                level_2_classes.add('控制计算')
            if label in yaokong_classes:
                level_2_classes.add('遥控')
            if label in yaoce_classes:
                level_2_classes.add('遥测')
            if label in xingnengxuqiu_classes:
                level_2_classes.add('性能需求')
            if label in kefuyongxuqiu_classes:
                level_2_classes.add('可复用需求')
            if label in keweihuxuqiu_classes:
                level_2_classes.add('可维护需求')
            if label in kekaoxingxuqiu_classes:
                level_2_classes.add('可靠性需求')
            if label in anquanxingxuqiu_classes:
                level_2_classes.add('安全性需求')
        level_2 = {'req': req, 'class': list(level_2_classes)}  
        result.append(level_2)
    return result

def process_level_2(true_path, pred_path, pred_json_path, hierarchy_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    pred_json = read_json(pred_json_path)
    hierarchy = read_json(hierarchy_path)
    
    assert len(true_df) == len(pred_df) == len(pred_json)-1
    
    reqs = true_df['req'].tolist()  
    
    chushihua_classes,shujucaiji_classes,kongzhishuchu_classes,kongzhijisuan_classes,yaokong_classes,yaoce_classes,xingnengxuqiu_classes,kefuyongxuqiu_classes,keweihuxuqiu_classes,kekaoxingxuqiu_classes,anquanxingxuqiu_classes = parse_hierarchy(hierarchy)
    
    true_level_2_df = map_to_level_2_df(true_df, reqs, chushihua_classes,shujucaiji_classes,kongzhishuchu_classes,kongzhijisuan_classes,yaokong_classes,yaoce_classes,xingnengxuqiu_classes,kefuyongxuqiu_classes,keweihuxuqiu_classes,kekaoxingxuqiu_classes,anquanxingxuqiu_classes)
    pred_level_2_df = map_to_level_2_df(pred_df, reqs, chushihua_classes,shujucaiji_classes,kongzhishuchu_classes,kongzhijisuan_classes,yaokong_classes,yaoce_classes,xingnengxuqiu_classes,kefuyongxuqiu_classes,keweihuxuqiu_classes,kekaoxingxuqiu_classes,anquanxingxuqiu_classes)
    
    pred_json_level_2 = map_to_level_2_json(pred_json, chushihua_classes,shujucaiji_classes,kongzhishuchu_classes,kongzhijisuan_classes,yaokong_classes,yaoce_classes,xingnengxuqiu_classes,kefuyongxuqiu_classes,keweihuxuqiu_classes,kekaoxingxuqiu_classes,anquanxingxuqiu_classes)
    
    true_level_2_df.to_excel(os.path.join(output_dir, 'true_level_2.xlsx'), index=False)
    pred_level_2_df.to_excel(os.path.join(output_dir, 'pred_level_2.xlsx'), index=False)
    save_json(pred_json_level_2, os.path.join(output_dir, 'pred_json_level_2.json'))

if __name__ == "__main__":
    true_path = r"path/to/true_labels,xlsx"
    pred_path = r"path/to/hier.xlsx"
    pred_json_path = r"path/to/hier.json"
    hierarchy_path = r"path/to/hierarchy.json"
    output_dir = r"path/to/level_2"
    process_level_2(true_path, pred_path, pred_json_path, hierarchy_path, output_dir)