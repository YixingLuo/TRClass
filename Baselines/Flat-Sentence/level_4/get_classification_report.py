import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
LABEL = ["故障诊断","数据有效性判断"]
def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def extract_labels(df, labels):
    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def compute_classification_report(y_true, y_pred, labels, output_path):
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=labels, 
        digits=4, 
        zero_division=0, 
        output_dict=False  
    )
    
    print(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("classification report:\n")
        f.write("=" * 50 + "\n")
        f.write(report)
        f.write("req："+str(len(y_true)))
    print(f"saved to {output_path}")

def process_classification_report(true_path, pred_path, output_path):
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    
    assert len(true_df) == len(pred_df)
    assert list(true_df.columns) == list(pred_df.columns)
    
    y_true = extract_labels(true_df, LABEL)
    y_pred = extract_labels(pred_df, LABEL)
    
    assert set(np.unique(y_true)).issubset({0, 1})
    assert set(np.unique(y_pred)).issubset({0, 1})
    
    compute_classification_report(y_true, y_pred, LABEL, output_path)

if __name__ == "__main__":
    true_path = r"path/to/true_level_4.xlsx" 
    pred_path = r"path/to/pred_level_4.xlsx" 
    output_path = r"path/to/classification_report.txt"  
    process_classification_report(true_path, pred_path, output_path)