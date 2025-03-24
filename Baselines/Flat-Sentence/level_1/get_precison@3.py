import pandas as pd
import numpy as np

LABEL = ["功能需求","非功能需求"]
def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def extract_labels(df, labels):

    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def compute_precision_at_3(y_true, y_pred):
    k = 3
    precisions = []
    
    for i in range(len(y_true)):
        true_labels = y_true[i]  
        pred_labels = y_pred[i]  
        
        assert set(np.unique(y_true)).issubset({0, 1})
        assert set(np.unique(y_pred)).issubset({0, 1})

        true_count = np.sum(true_labels)  
        pred_count = np.sum(pred_labels)  
        correct = np.sum(true_labels * pred_labels)  
        
        # min(k, |C_i^pred|, |C_i^true|)
        denominator = min(k, true_count) if true_count > 0 else 1
        
        precision = correct / denominator if denominator > 0 else 0
        precisions.append(precision)

    mean_precision = np.mean(precisions) if precisions else 0
    return mean_precision, precisions

def compute_precision_at_3_from_tables(true_path, pred_path, output_path="path/to/precision_at_3.txt"):

    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    
    assert len(true_df) == len(pred_df)
    assert list(true_df.columns) == list(pred_df.columns)

    y_true = extract_labels(true_df, LABEL)
    y_pred = extract_labels(pred_df, LABEL)

    precision_at_3, precisions = compute_precision_at_3(y_true, y_pred)

    print(f"Precision@3: {precision_at_3:.4f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Precision@3: {precision_at_3:.4f}\n")
        f.write("\n Precision@3 of every sample：\n")
        for i, precision in enumerate(precisions):
            true_labels_set = set(LABEL[j] for j in np.where(y_true[i] == 1)[0])
            pred_labels_set = set(LABEL[j] for j in np.where(y_pred[i] == 1)[0])
            intersection = true_labels_set & pred_labels_set
            f.write(f"sample {i+1}: Precision@3 = {precision:.4f}, true lables: {true_labels_set}, predict lables: {pred_labels_set}, intersection: {intersection}\n")
    print(f"saved to {output_path}")

if __name__ == "__main__":
    true_path = r"path/to/true_level_1.xlsx"  
    pred_path = r"path/to/pred_level_1.xlsx"  
    output_path = r"path/to/precision_at_3.txt"  
    compute_precision_at_3_from_tables(true_path, pred_path, output_path=output_path)