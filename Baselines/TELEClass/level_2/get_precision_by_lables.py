import pandas as pd
import os

def read_table(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return None

def compute_accuracy_metrics(true_path, pred_path, output_path="path/to/precision_by_labels.txt"):
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    
    if true_df is None or pred_df is None:
        return
    
    if len(true_df) != len(pred_df):
        return
    
    if list(true_df.columns) != list(pred_df.columns):
        return
    
    if not (true_df['req'] == pred_df['req']).all():
        return
    
    labels = ["初始化","数据采集","控制输出","控制计算","遥控","遥测","性能需求","可复用需求","可维护需求","可靠性需求","安全性需求"]
    metrics = {}
    total_samples = len(true_df)
    predict_counts={}
    for label in labels:
        predict_positive_reqs = pred_df[pred_df[label] == 1]['req'].tolist()
        predict_counts[label]=len(predict_positive_reqs)
        
        tp = 0
        for req in predict_positive_reqs:
            true_value = true_df[true_df['req'] == req][label].iloc[0]
            if true_value == 1:
                tp += 1
        
        if predict_counts[label] > 0:
            tp_rate = tp / predict_counts[label]
        else:
            tp_rate = 0.0  
        
        metrics[label] = {
            'TP': tp,
            'Predict Count': predict_counts[label],
            'TP Rate': tp_rate
        }
    
    simple_average = sum(metric['TP Rate'] for metric in metrics.values()) / len(metrics)
    
    total_predict_samples = sum(predict_counts.values())
    weighted_average = 0.0
    if total_predict_samples > 0:
        weighted_average = sum(metric['TP Rate'] * predict_counts[label] for label, metric in metrics.items()) / total_predict_samples
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for label, metric in metrics.items():
            f.write(f"{label}:\n")
            f.write(f"  TP: {metric['TP']}\n")
            f.write(f"  predict Count: {metric['Predict Count']}\n")
            f.write(f"  TP Rate: {metric['TP Rate']:.4f}\n")
        f.write("\n")
        f.write(f"simple_average TP Rate: {simple_average:.4f}\n")
        f.write(f"weighted_average TP Rate: {weighted_average:.4f}\n")
        f.write(f"total samples: {total_samples}\n")
    
    print(f"saved to {output_path}")
# 使用示例
if __name__ == "__main__":
    true_path = r"path/to/true_level_2.xlsx"
    pred_path = r"path/to/pred_level_2.xlsx"
    output_path = r"path/to/precision_by_labels.txt"
    compute_accuracy_metrics(true_path, pred_path, output_path)