import pandas as pd
import numpy as np

LABEL = ["功能需求","非功能需求"]

def read_table(file_path):
    df = pd.read_excel(file_path)
    return df

def extract_labels(df, labels):
    label_data = df[labels].to_numpy()  #  (n_samples, n_labels)
    return label_data

def compute_example_f1(y_true, y_pred):
    precisions = []
    
    for i in range(len(y_true)):
        true_labels = y_true[i]  
        pred_labels = y_pred[i]  
        
        assert set(np.unique(y_true)).issubset({0, 1})
        assert set(np.unique(y_pred)).issubset({0, 1})

        true_count = np.sum(true_labels)  # |C_i^{true}|
        

        pred_count = np.sum(pred_labels)  # |C_i^{pred}|
        

        intersection = np.sum(true_labels * pred_labels)  # |C_i^{true} ∩ C_i^{pred}|
        

        numerator = 2 * intersection
        
        # |C_i^{true}| + |C_i^{pred}|
        denominator = true_count + pred_count

        if denominator == 0:
            precision = 0.0
        else:
            precision = numerator / denominator if numerator > 0 else 0.0
        precisions.append(precision)
    
    mean_f1 = np.mean(precisions) if precisions else 0
    return mean_f1, precisions
def compute_example_f1_from_tables(true_path, pred_path, output_path="path/to/example_f1.txt"):
    true_df = read_table(true_path)
    pred_df = read_table(pred_path)
    
    assert len(true_df) == len(pred_df)
    assert list(true_df.columns) == list(pred_df.columns)
    
    y_true = extract_labels(true_df, LABEL)
    y_pred = extract_labels(pred_df, LABEL)
    
    example_f1, f1_scores = compute_example_f1(y_true, y_pred)
    
    print(f"Example-F1: {example_f1:.4f}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Example-F1: {example_f1:.4f}\n")
        f.write("\n Example-F1：\n")
        for i, f1_score in enumerate(f1_scores):
            true_labels_set = set(LABEL[j] for j in np.where(y_true[i] == 1)[0])
            pred_labels_set = set(LABEL[j] for j in np.where(y_pred[i] == 1)[0])
            intersection = true_labels_set & pred_labels_set
            f.write(f"sample {i+1}: Example-F1 = {f1_score:.4f}, true labels: {true_labels_set}, predict labels: {pred_labels_set}, intersection: {intersection}\n")
    print(f"saved to {output_path}")

if __name__ == "__main__":
    true_path = r"path/to/true_level_1.xlsx"  
    pred_path = r"path/to/pred_level_1.xlsx"  
    output_path = r"path/to/example_f1.txt"  
    compute_example_f1_from_tables(true_path, pred_path, output_path=output_path)