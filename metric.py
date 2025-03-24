import json
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt

def get_labels_from_row(row):
    labels = []
    for i in range(1, 3):  
        level_path = []
        for j in range(1, 5):  
            level = row[f'Level{j}_{i}']
            if pd.notna(level):
                level_path.append(level)
        if level_path:
            labels.append('-'.join(level_path))
    return labels
 
def get_labels_from_row_single(row):
    labels = []
    for i in range(1, 2):  
        level_path = []
        for j in range(1, 5):
            level = row[f'Level{j}_{i}']
            if pd.notna(level):
                level_path.append(level)
        if level_path:
            labels.append('-'.join(level_path))
    return labels

import json

def load_predictions_from_answer_jsonl(file_path):
    
    predictions = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data.get("conversations", [])
            last_gpt_label = None
            for convo in conversations:
                if convo['from'] == 'gpt':
                    last_gpt_label = convo['value']
            
            if last_gpt_label:
                predictions.append([{last_gpt_label: 1.0}])

    return predictions
def load_predictions_from_answer_jsonl_v2(file_path):
    predictions = []
    confidence_pattern = re.compile(r'(.+?)\s*\(conf=([\d\.]+)\)')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data.get("conversations", [])
            
            last_gpt_value = None
            for convo in conversations:
                if convo['from'] == 'gpt':
                    last_gpt_value = convo['value']
            if isinstance(last_gpt_value, list):
                label_conf_list = []
                for label_with_conf in last_gpt_value:
                    match = confidence_pattern.match(label_with_conf)
                    if match:
                        label = match.group(1).strip()
                        confidence = float(match.group(2))
                        label_conf_list.append({label: confidence})
                predictions.append(label_conf_list)
            elif isinstance(last_gpt_value, str):
                predictions.append([{last_gpt_value: 1.0}])

    return predictions

def compute_metrics(predictions, ground_truth_file, k=3):
    if ground_truth_file.endswith('.json'):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for item in data:
            example = item.get('Example', {})
            description = example.get('Description', '').strip()
            labels = example.get('Correct answer', [])
            labels = [label for label in labels if label]  

            records.append({
                'Requirements Description': description,
                'Labels': labels
            })

        ground_truth = pd.DataFrame(records)

    elif ground_truth_file.endswith('.xlsx'):
        ground_truth = pd.read_excel(ground_truth_file)
        ground_truth['Labels'] = ground_truth.apply(get_labels_from_row, axis=1)

    num_samples = min(len(predictions), len(ground_truth))

    sample_metrics = []
    example_f1_list = []

    label_stats = {}  

    def ensure_label_dict(label):
        if label not in label_stats:
            label_stats[label] = {'tp': 0, 'fp': 0, 'fn': 0}
        return label_stats[label]

    for idx in range(num_samples):
        pred_list = predictions[idx]
        pred_scores = {}
        for p in pred_list:
            for label, score in p.items():
                cleaned_label = label.replace('root-', '')
                pred_scores[cleaned_label] = score

        sorted_preds = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_preds = [x[0] for x in sorted_preds[:k]]

        prediction_count = len(top_k_preds)
        true_labels = ground_truth.iloc[idx]['Labels']
        true_label_count = len(true_labels)

        overlap_top1 = len(set(top_k_preds[:1]) & set(true_labels))
        precision_at_1 = overlap_top1 / min(1, true_label_count) if true_label_count > 0 else 0

        overlap_top3 = len(set(top_k_preds) & set(true_labels))
        precision_at_3 = overlap_top3 / min(3, true_label_count) if true_label_count > 0 else 0

        recall = overlap_top3 / true_label_count if true_label_count > 0 else 0

        precision = overlap_top3 / prediction_count if prediction_count > 0 else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        pred_set = set(top_k_preds)
        true_set = set(true_labels)
        intersection = len(pred_set & true_set)
        union = len(pred_set) + len(true_set)
        example_f1 = 2 * intersection / union if union > 0 else 0
        example_f1_list.append(example_f1)

        reciprocal_ranks = []
        for t_label in true_labels:
            if t_label in top_k_preds:
                rank = top_k_preds.index(t_label) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        sample_mrr = sum(reciprocal_ranks) / len(true_labels) if true_label_count > 0 else 0

        sample_metrics.append({
            "sample_id": idx + 1,
            "precision@1": precision_at_1,
            "precision@3": precision_at_3,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "example_f1": example_f1,
            "MRR": sample_mrr
        })

        for label in pred_set:
            if label in true_set:
                ensure_label_dict(label)['tp'] += 1
            else:
                ensure_label_dict(label)['fp'] += 1

        for label in true_set:
            if label not in pred_set:
                ensure_label_dict(label)['fn'] += 1

    df_metrics = pd.DataFrame(sample_metrics)
    print(df_metrics)

    print("\nOverall Metrics (Sample-level Average across samples):")
    print(f"Precision@1 (sample avg): {df_metrics['precision@1'].mean():.4f}")
    print(f"Precision@3 (sample avg): {df_metrics['precision@3'].mean():.4f}")
    print(f"Precision (sample avg):    {df_metrics['precision'].mean():.4f}")
    print(f"Recall (sample avg):       {df_metrics['recall'].mean():.4f}")
    print(f"F1-score (sample avg):     {df_metrics['f1'].mean():.4f}")
    print(f"Example-F1 (sample avg):   {sum(example_f1_list) / len(example_f1_list):.4f}")
    print(f"MRR (sample avg):          {df_metrics['MRR'].mean():.4f}")

    all_labels = list(label_stats.keys())
    label_precision = {}
    label_recall = {}
    label_f1 = {}
    label_support = {}  

    for label in all_labels:
        tp = label_stats[label]['tp']
        fp = label_stats[label]['fp']
        fn = label_stats[label]['fn']
        # P/R/F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        label_precision[label] = p
        label_recall[label] = r
        label_f1[label] = f
        label_support[label] = (tp + fn)  

    n_labels = len(all_labels)
    macro_p = sum(label_precision.values()) / n_labels if n_labels > 0 else 0
    macro_r = sum(label_recall.values()) / n_labels if n_labels > 0 else 0
    macro_f = sum(label_f1.values()) / n_labels if n_labels > 0 else 0

    total_support = sum(label_support.values())
    if total_support > 0:
        weighted_p = sum(label_precision[l] * label_support[l] for l in all_labels) / total_support
        weighted_r = sum(label_recall[l] * label_support[l] for l in all_labels) / total_support
        weighted_f = sum(label_f1[l] * label_support[l] for l in all_labels) / total_support
    else:
        weighted_p = weighted_r = weighted_f = 0.0

    print("\nLabel-level Metrics (Macro / Weighted):")
    print(f"Macro Precision:  {macro_p:.4f}")
    print(f"Macro Recall:     {macro_r:.4f}")
    print(f"Macro F1:         {macro_f:.4f}")
    print(f"Weighted Precision:  {weighted_p:.4f}")
    print(f"Weighted Recall:     {weighted_r:.4f}")
    print(f"Weighted F1:         {weighted_f:.4f}")


    n_values = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]

    metrics_results = {
        'n': [],
        'precision@1': [],
        'precision@3': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'example_f1': [],
        'mrr': []
    }

    for i in range(1, len(n_values)):
        start = n_values[i - 1]
        end = n_values[i]
        subset = df_metrics.iloc[start:end]  # [start, end)

        n = n_values[i]
        metrics_results['n'].append(n)
        metrics_results['precision@1'].append(subset['precision@1'].mean() if len(subset) > 0 else 0)
        metrics_results['precision@3'].append(subset['precision@3'].mean() if len(subset) > 0 else 0)
        metrics_results['precision'].append(subset['precision'].mean() if len(subset) > 0 else 0)
        metrics_results['recall'].append(subset['recall'].mean() if len(subset) > 0 else 0)
        metrics_results['f1'].append(subset['f1'].mean() if len(subset) > 0 else 0)

        if end <= len(example_f1_list):
            metrics_results['example_f1'].append(
                sum(example_f1_list[start:end]) / (end - start) if (end - start) > 0 else 0
            )
        else:
            real_end = min(end, len(example_f1_list))
            real_count = real_end - start
            if real_count > 0:
                metrics_results['example_f1'].append(
                    sum(example_f1_list[start:real_end]) / real_count
                )
            else:
                metrics_results['example_f1'].append(0)

        metrics_results['mrr'].append(subset['MRR'].mean() if len(subset) > 0 else 0)

    print("\nMetrics by Sample Size (in segments):")
    for i, n in enumerate(metrics_results['n']):
        print(f"\n--- n = {n} ---")
        print(f"Precision@1: {metrics_results['precision@1'][i]:.4f}")
        print(f"Precision@3: {metrics_results['precision@3'][i]:.4f}")
        print(f"Precision:   {metrics_results['precision'][i]:.4f}")
        print(f"Recall:      {metrics_results['recall'][i]:.4f}")
        print(f"F1-score:    {metrics_results['f1'][i]:.4f}")
        print(f"Example-F1:  {metrics_results['example_f1'][i]:.4f}")
        print(f"MRR:         {metrics_results['mrr'][i]:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_results['n'], metrics_results['precision@1'], marker='o', label='Precision@1')
    plt.plot(metrics_results['n'], metrics_results['precision@3'], marker='s', label='Precision@3')
    plt.plot(metrics_results['n'], metrics_results['precision'], marker='v', label='Precision')
    plt.plot(metrics_results['n'], metrics_results['recall'], marker='^', label='Recall')
    plt.plot(metrics_results['n'], metrics_results['f1'], marker='d', label='F1-score')
    plt.plot(metrics_results['n'], metrics_results['example_f1'], marker='*', label='Example-F1')
    plt.plot(metrics_results['n'], metrics_results['mrr'], marker='p', label='MRR')

    plt.title("Metric Trends by Sample Size")
    plt.xlabel("Number of Samples")
    plt.ylabel("Score")
    plt.xticks(metrics_results['n'])
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def print_prediction_stats(predictions):
    label_counter = Counter()
    
    for pred_list in predictions:
        for p in pred_list:
            for label in p.keys():
                label_counter[label] += 1
    
    print("\nPrediction Label Distribution:")
    for label, count in label_counter.most_common():
        print(f"{label}: {count}")

def split_path(path_str):
    parts = path_str.split('-')
    while len(parts) < 4:
        parts.append('')
    return parts[0], parts[1], parts[2], parts[3]


def get_labels_from_row(row):
    labels = []
    for i in range(1, 3): 
        level_path = []
        for j in range(1, 5): 
            level = row[f'Level{j}_{i}']
            if pd.notna(level):
                level_path.append(level)
        if level_path:
            labels.append('-'.join(level_path))
    return labels


import json
import pandas as pd
from collections import defaultdict

def split_path(path_str):
    parts = path_str.split('-')
    while len(parts) < 4:
        parts.append('')
    return parts[0], parts[1], parts[2], parts[3]

def get_labels_from_row(row):
    if isinstance(row['Labels'], str):
        labels = [lab.strip() for lab in row['Labels'].split('\n') if lab.strip()]
    elif isinstance(row['Labels'], list):
        labels = row['Labels']
    else:
        labels = []
    return labels

def compute_metrics_by_level_2(predictions, ground_truth_file, k=3):
    if ground_truth_file.endswith('.json'):
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = []
        for item in data:
            example = item.get('Example', {})
            description = example.get('Description', '').strip()
            labels = example.get('Correct answer', [])
            labels = [label for label in labels if label]  
            records.append({
                'Requirements Description': description,
                'Labels': labels
            })
        df = pd.DataFrame(records)
    else:
        df = pd.read_excel(ground_truth_file)
        df['Labels'] = df.apply(get_labels_from_row, axis=1)

    p1_level   = {1: [], 2: [], 3: [], 4: []}  # Precision@1
    p3_level   = {1: [], 2: [], 3: [], 4: []}  # Precision@3
    mrr_level  = {1: [], 2: [], 3: [], 4: []}  # MRR
    f1_level   = {1: [], 2: [], 3: [], 4: []}  # Example-F1 (Dice)
    
    p_level    = {1: [], 2: [], 3: [], 4: []}  # Sample-level Precision
    r_level    = {1: [], 2: [], 3: [], 4: []}  # Sample-level Recall
    f1sample_level = {1: [], 2: [], 3: [], 4: []}  # Sample-level F1

    macro_tp = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
    macro_pred = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
    macro_all_gt = {1: set(), 2: set(), 3: set(), 4: set()}
    macro_gt_count = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}

    num_samples = min(len(predictions), len(df))

    for idx in range(num_samples):
        true_paths = df.iloc[idx]['Labels']  
        ground_levels = {1: set(), 2: set(), 3: set(), 4: set()}
        for path_str in true_paths:
            l1, l2, l3, l4 = split_path(path_str)
            if l1: ground_levels[1].add(l1)
            if l2: ground_levels[2].add(l2)
            if l3: ground_levels[3].add(l3)
            if l4: ground_levels[4].add(l4)
        
        pred_list = predictions[idx]  # [{p1: conf1}, {p2: conf2}, ...]
        pred_scores = {}
        for p in pred_list:
            for label, score in p.items():
                cleaned_label = label.replace('root-', '')
                pred_scores[cleaned_label] = score
        sorted_preds = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_paths = [item[0] for item in sorted_preds[:k]]
        
        top1_levels = {1: set(), 2: set(), 3: set(), 4: set()}
        if len(top_k_paths) > 0:
            l1, l2, l3, l4 = split_path(top_k_paths[0])
            if l1: top1_levels[1].add(l1)
            if l2: top1_levels[2].add(l2)
            if l3: top1_levels[3].add(l3)
            if l4: top1_levels[4].add(l4)
        
        top3_levels = {1: set(), 2: set(), 3: set(), 4: set()}
        for i in range(min(k, len(top_k_paths))):
            l1, l2, l3, l4 = split_path(top_k_paths[i])
            if l1: top3_levels[1].add(l1)
            if l2: top3_levels[2].add(l2)
            if l3: top3_levels[3].add(l3)
            if l4: top3_levels[4].add(l4)
        
        for level_i in range(1, 5):
            # Precision@1
            if top1_levels[level_i].intersection(ground_levels[level_i]):
                p1_level[level_i].append(1.0)
            else:
                p1_level[level_i].append(0.0)
            
            # Precision@3
            if top3_levels[level_i].intersection(ground_levels[level_i]):
                p3_level[level_i].append(1.0)
            else:
                p3_level[level_i].append(0.0)
            
            # MRR
            g_labels = list(ground_levels[level_i])
            if len(g_labels) == 0:
                mrr_level[level_i].append(0.0)
            else:
                reciprocal_ranks = []
                for g in g_labels:
                    rank_found = 0
                    for rank_idx, path_str in enumerate(top_k_paths):
                        preds = split_path(path_str)
                        predict_label = preds[level_i - 1]
                        if predict_label == g:
                            rank_found = rank_idx + 1
                            break
                    if rank_found > 0:
                        reciprocal_ranks.append(1.0 / rank_found)
                    else:
                        reciprocal_ranks.append(0.0)
                mrr_level[level_i].append(sum(reciprocal_ranks) / len(reciprocal_ranks))
            
            # Example-F1 (Dice)
            inter_size = len(top3_levels[level_i].intersection(ground_levels[level_i]))
            union_size = len(top3_levels[level_i]) + len(ground_levels[level_i])
            if union_size > 0:
                dice = 2.0 * inter_size / union_size
            else:
                dice = 0.0
            f1_level[level_i].append(dice)
            
            # Sample-level Precision, Recall, F1 
            pred_set = top3_levels[level_i]
            true_set = ground_levels[level_i]
            inter_size = len(pred_set & true_set)
            p_val = inter_size / len(pred_set) if len(pred_set) > 0 else 0.0
            r_val = inter_size / len(true_set) if len(true_set) > 0 else 0.0
            f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
            p_level[level_i].append(p_val)
            r_level[level_i].append(r_val)
            f1sample_level[level_i].append(f1_val)

            for label in top3_levels[level_i]:
                macro_pred[level_i][label] += 1
                if label in ground_levels[level_i]:
                    macro_tp[level_i][label] += 1
            
            for label in ground_levels[level_i]:
                macro_all_gt[level_i].add(label)
                macro_gt_count[level_i][label] += 1

    print("\n=== Level-wise Metrics (over {} samples) ===".format(num_samples))
    for level_i in range(1, 5):
        p1_avg  = sum(p1_level[level_i]) / num_samples
        p3_avg  = sum(p3_level[level_i]) / num_samples
        mrr_avg = sum(mrr_level[level_i]) / num_samples
        dice_avg  = sum(f1_level[level_i]) / num_samples  # Example-F1 (Dice)
        p_avg = sum(p_level[level_i]) / num_samples
        r_avg = sum(r_level[level_i]) / num_samples
        f1_avg = sum(f1sample_level[level_i]) / num_samples

        print(f"[Level {level_i}] "
              f"Precision@1={p1_avg:.4f}, Precision@3={p3_avg:.4f}, MRR={mrr_avg:.4f}, "
              f"Example-F1(Dice)={dice_avg:.4f}, Sample-P={p_avg:.4f}, Sample-R={r_avg:.4f}, Sample-F1={f1_avg:.4f}")

    print("\n=== Macro-Precision per Level ===")
    for level_i in range(1, 5):
        all_labels = set(macro_pred[level_i].keys()).union(macro_all_gt[level_i])
        macro_precisions = []
        for label in all_labels:
            if macro_pred[level_i][label] > 0:
                precision = macro_tp[level_i][label] / macro_pred[level_i][label]
            else:
                precision = 0.0
            macro_precisions.append(precision)
        macro_precision_avg = sum(macro_precisions) / len(all_labels) if all_labels else 0.0
        print(f"[Level {level_i}] Macro-Precision={macro_precision_avg:.4f}")

    print("\n=== Weighted Macro-Precision per Level ===")
    for level_i in range(1, 5):
        all_labels = set(macro_pred[level_i].keys()).union(macro_all_gt[level_i])
        total_count = sum(macro_gt_count[level_i][lab] for lab in all_labels)
        if total_count == 0:
            weighted_macro_precision = 0.0
        else:
            weighted_sum = 0.0
            for label in all_labels:
                if macro_pred[level_i][label] > 0:
                    label_precision = macro_tp[level_i][label] / macro_pred[level_i][label]
                else:
                    label_precision = 0.0
                weighted_sum += label_precision * macro_gt_count[level_i][label]
            weighted_macro_precision = weighted_sum / total_count
        print(f"[Level {level_i}] Weighted Macro-Precision={weighted_macro_precision:.4f}")

    print("\n=== Weighted Metrics per Level (based on GT frequency) ===")
    for level_i in range(1, 5):
        all_labels = macro_all_gt[level_i]
        total_gt = sum(macro_gt_count[level_i][lab] for lab in all_labels)
        if total_gt == 0:
            weighted_precision = 0.0
            weighted_recall = 0.0
            weighted_f1 = 0.0
        else:
            sum_precision = 0.0
            sum_recall = 0.0
            sum_f1 = 0.0
            for lab in all_labels:
                tp = macro_tp[level_i][lab]
                pred_count = macro_pred[level_i][lab]
                gt_count = macro_gt_count[level_i][lab]
                prec = tp / pred_count if pred_count > 0 else 0.0
                rec = tp / gt_count if gt_count > 0 else 0.0
                f1score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                sum_precision += prec * gt_count
                sum_recall += rec * gt_count
                sum_f1 += f1score * gt_count
            weighted_precision = sum_precision / total_gt
            weighted_recall = sum_recall / total_gt
            weighted_f1 = sum_f1 / total_gt
        print(f"[Level {level_i}] Weighted Precision={weighted_precision:.4f}, "
              f"Weighted Recall={weighted_recall:.4f}, Weighted F1={weighted_f1:.4f}")

    print("\n=== Labels per Level (Ground Truth) ===")
    for level_i in range(1, 5):
        labels = sorted(list(macro_all_gt[level_i]))
        print(f"[Level {level_i}] Labels: {labels}")


if __name__ == '__main__':


    predictions = load_predictions_from_answer_jsonl_v2('answer_multi_total_gpt-3.5-turbo.json') ##baseline
    print_prediction_stats(predictions)

    compute_metrics(predictions, 'selected_samples.json', k=3)
    compute_metrics_by_level_2(
        predictions,
        ground_truth_file='selected_samples.json',
        k=3
    )

