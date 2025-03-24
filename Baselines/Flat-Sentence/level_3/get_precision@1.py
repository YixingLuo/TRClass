import json
import pandas as pd
import numpy as np
from typing import List, Dict
LABEL = ["姿态控制","姿态确定","运行时保障","模式管理","轨道计算","精度约束","空间约束","时间约束"]
def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def read_json(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return []

def extract_labels(df, labels):
    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def compute_precision_at_1(y_true, pred_classes):
    precisions = []
    
    for i in range(len(y_true)):
        true_labels = y_true[i] 
        pred_class = pred_classes[i][0] if pred_classes[i] else None  
        assert set(np.unique(y_true)).issubset({0, 1})
        
        true_count = np.sum(true_labels)  
        pred_label = np.zeros(len(LABEL))
        if pred_class in LABEL:
            pred_label[LABEL.index(pred_class)] = 1
        
        correct = np.sum(true_labels * pred_label)  
        denominator = min(1, true_count) if true_count > 0 else 1

        precision = correct / denominator if denominator > 0 else 0
        precisions.append(precision)
   
    mean_precision = np.mean(precisions) if precisions else 0
    return mean_precision, precisions

def compute_precision_at_1_from_tables_and_json(true_path, pred_json_path, output_path="precision_at_1.txt"):

    true_df = read_table(true_path)
    
    pred_data = read_json(pred_json_path)
    
    assert len(true_df) == len(pred_data)
    y_true = extract_labels(true_df, LABEL)

    pred_classes = [item['class'] for item in pred_data]
    
    precision_at_1, precisions = compute_precision_at_1(y_true, pred_classes)
    
    print(f"Precision@1: {precision_at_1:.4f}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Precision@1: {precision_at_1:.4f}\n")
        f.write("\n Precision@1 of every sample：\n")
        for i, precision in enumerate(precisions):
            true_labels_set = set(LABEL[j] for j in np.where(y_true[i] == 1)[0])
            pred_first = pred_classes[i][0] if pred_classes[i] else "none"
            pred_labels_set = {pred_first} if pred_classes[i] else set()
            intersection = true_labels_set & pred_labels_set
            f.write(f"sample {i+1}: Precision@1 = {precision:.4f}, true lables: {true_labels_set}, predict lable: {pred_labels_set}, intersection: {intersection}\n")
    print(f"saved to {output_path}")

if __name__ == "__main__":
    true_path = r"path/to/true_level_3.xlsx" 
    pred_json_path = r"path/to/pred_json_level_3.json" 
    output_path = r"path/to/flatten_precision_at_1.txt"  
    compute_precision_at_1_from_tables_and_json(true_path, pred_json_path, output_path=output_path)