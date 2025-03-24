import pandas as pd
import numpy as np
import json

LABEL = ["故障诊断","数据有效性判断"]

def read_true_table(file_path):
    df = pd.read_excel(file_path)
    return df

def read_predict_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_true_labels(df, labels):
    label_data = df[labels].to_numpy()  # (n_samples, n_labels)
    return label_data

def get_predicted_ranking(pred_labels, all_labels):
    ranked_labels = pred_labels[:3]  
    remaining_labels = [label for label in all_labels if label not in ranked_labels]
    ranked_labels.extend(remaining_labels)

    assert len(ranked_labels) == len(all_labels)
    assert set(ranked_labels) == set(all_labels)
    
    return ranked_labels

def compute_mrr(y_true, pred_data, all_labels):
    reciprocal_ranks = []
    
    for i in range(len(y_true)):
        true_labels = y_true[i]  
        pred_item = pred_data[i]  
        
        true_label_set = set(all_labels[j] for j in np.where(true_labels == 1)[0])
        
        if not true_label_set:
            reciprocal_ranks.append(0)
            continue

        pred_labels = pred_item["class"]
        
        sample_reciprocal_ranks = []
        for true_label in true_label_set:
            rank = 0
            isFlag=False
            for pred_label in pred_labels:
                rank += 1
                if pred_label == true_label:
                    sample_reciprocal_ranks.append(1 / rank)
                    isFlag=True
                    break
            if not isFlag:
                sample_reciprocal_ranks.append(0)

        sample_mrr = np.mean(sample_reciprocal_ranks) if sample_reciprocal_ranks else 0
        reciprocal_ranks.append(sample_mrr)
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    return mrr, reciprocal_ranks

def compute_mrr_from_data(true_path, pred_path, output_path="mrr.txt"):
    true_df = read_true_table(true_path)
    
    pred_data = read_predict_json(pred_path)
    
    assert len(true_df) == len(pred_data)
    
    y_true = extract_true_labels(true_df, LABEL)
    
    assert set(np.unique(y_true)).issubset({0, 1})
    true_reqs = true_df["req"].tolist()
    pred_reqs = [item["req"] for item in pred_data]
    assert true_reqs == pred_reqs
    mrr, reciprocal_ranks = compute_mrr(y_true, pred_data, LABEL)
    
    print(f"MRR: {mrr:.4f}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"MRR: {mrr:.4f}\n")
        f.write("\n Reciprocal Rank of every sample：\n")
        for i in range(len(reciprocal_ranks)):
            true_labels_set = set(LABEL[j] for j in np.where(y_true[i] == 1)[0])
            pred_labels = pred_data[i]["class"]
            f.write(f"sample {i+1}: Reciprocal Rank = {reciprocal_ranks[i]:.4f}, true lables: {true_labels_set}, predict lables: {pred_labels}\n")
    print(f"saved to {output_path}")

if __name__ == "__main__":
    true_path = r"path/to/true_level_4.xlsx"  
    pred_path = r"path/to/pred_json_level_4.json"  
    output_path = r"path/to/mrr.txt"
    compute_mrr_from_data(true_path, pred_path, output_path=output_path)