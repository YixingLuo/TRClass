import sys
import numpy as np
import argparse
import json
import os
from threading import Thread
import re
import random
import torch
from transformers import AutoTokenizer, AutoModel
import jieba
from rouge_chinese import Rouge
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

def load_tree_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data 

def find_node_by_path(tree, path):
    current_node = tree
    for part in path[1:]:
        found = False
        for child in current_node.get('children', []):
            if child.get('name') == part:
                current_node = child
                found = True
                break
        if not found:
            return None
    return current_node
def get_child_node(file_name, father):
    base_tree = load_tree_from_json(file_name)
    father_list = father.split('-')
    father_node = find_node_by_path(base_tree, father_list)
    if father_node is None:
        return []
    children_node = father_node.get('children', [])
    children_node = [child.get('name') for child in children_node]
    return children_node

def my_cosine_similarity(a,b):
    answer=0
    n=len(a)
    for i in range(n):
        answer+=a[i]*b[i]
    return answer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-zh-v1.5")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(device)
reranker_model.eval()
def get_embedding_bge(text):
    candidate_inputs = tokenizer(text, padding=True, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        candidate_outputs = model(**candidate_inputs)
        candidate_embedding = candidate_outputs.last_hidden_state[:, 0]
        candidate_embedding = torch.nn.functional.normalize(candidate_embedding, p=2, dim=1)
    return candidate_embedding.cpu().numpy()[0]

def tokenize_chinese_english(sentence):
    if isinstance(sentence, dict):  # Ensure we have a string
        sentence = sentence.get("Example", {}).get("Description", "")
    return ' '.join(jieba.cut(sentence))
def compute_rouge_multiple3(references, candidate):
    candidate_tokens = tokenize_chinese_english(candidate)

    max_similarity = float('-inf')
    max_index = 0

    mid_similarity = float('-inf')
    mid_index = 0

    min_similarity = float('-inf')
    min_index = 0

    for i, ref in enumerate(references):
        reference_tokens = tokenize_chinese_english(ref)
        rouge = Rouge()
        score = rouge.get_scores(candidate_tokens, reference_tokens)[0]['rouge-l']['f']
        if score > max_similarity:
            max_similarity = score
            max_index = i
        elif score > mid_similarity:
            mid_similarity = score
            mid_index = i
        elif score > min_similarity:
            min_similarity = score
            min_index = i

    return [max_similarity, mid_similarity, min_similarity], [max_index, mid_index, min_index]

def compute_cosine_multiple3(embedding_finally, embedding):
    dot_products = np.dot(embedding_finally, embedding)

    norms_finally = np.linalg.norm(embedding_finally, axis=1)

    norm_embedding = np.linalg.norm(embedding)

    cosine_similarities = dot_products / (norms_finally * norm_embedding)
    
    top3_indices = np.argsort(cosine_similarities)[::-1][:3]
    return top3_indices


def get_example(memory_finally,s, embedding_finally, merged_tree):
    if not memory_finally:
        return ""
    embedding = get_embedding_bge(s)

    index_list = compute_cosine_multiple3(embedding_finally, embedding)

    answer = ""
    for i in range(len(index_list)):
        answer += "Example " + str(i) + "\\n"
        max_index = index_list[i]
        father = "root"
        ans_list = memory_finally[max_index]["Correct answer"][0].split('-')
        final_ans = ""
        for item in ans_list:
            child_node = get_child_node(merged_tree, father)
            child_node_str = ""
            for idx in range(len(child_node)):
                select = chr(65 + idx) + ". "  
                child_node_str += select + child_node[idx] + "\\n"
            final_ans += "\\nOptions: \\n"
            final_ans += child_node_str
            for idx in range(len(child_node)):
                if child_node[idx] == item:
                    final_ans += "Answer: " + chr(65 + idx) + ". " + item
                    break
            father = father + "-" + item 
        answer += "Requirement Description:"+memory_finally[max_index]["Description"]+ "\\n Answer: " + memory_finally[max_index]["Correct answer"][0]+ "\\n"

    return answer

def get_example_from_sampled_data(sampled_data, sampled_embeddings, s, top_k=3):
    """Retrieve the top_k most similar examples from sampled_data.json."""
    if not sampled_data:
        return ""
    
    embedding = get_embedding_bge(s)
    scores = []
    
    for idx, entry in enumerate(sampled_data):
        example_text = entry.get("Example", {}).get("Description", "")
        example_embedding = sampled_embeddings[idx]
        
        score = np.dot(embedding, example_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(example_embedding)
        )
        
        scores.append((entry, score))
    
    top_matches = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    

    example_texts = []
    for match in top_matches:
        example_texts.append(
            f"Requirement Description: {match[0]['Example']['Description']}\n"
            f"Answer: {','.join([ans for ans in match[0]['Example']['Correct answer'] if ans])}\n"
        )
    
    return "\n".join(example_texts)

def extract_json(response_content):
    json_block_pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(json_block_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    json_pattern = r"(\{.*?\}|\[.*?\])"
    matches = re.findall(json_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    return None

LLM_MODEL = "qwq-32b"
def call_model_api(messages: List[Dict[str, str]]):
    client = OpenAI(
        api_key="", ## qwen
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if LLM_MODEL != "qwq-32b":
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            timeout=60
        )
        if '</think>' in response.choices[0].message.content:
            response_content = response.choices[0].message.content.split('</think>')[-1]
            print(f'llm: {response_content}')
            return response_content
        else:
            response_content = response.choices[0].message.content
            print(f'llm: {response_content}')
            return response_content
    else:
        response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=0,
                timeout=60,
                stream=True
            )
        reasoning_content = ''
        content = ''
        is_answering = False
        for chunk in response:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "content" + "=" * 20 + "\n")
                        is_answering = True
                    print(delta.content, end='', flush=True)
                    content += delta.content
        return content

def parse_multiple_answers(generated_text: str, child_node: List[str]) -> List[Tuple[str, float]]:
    text_lower = generated_text.lower()
    answer_pattern = re.search(r"answer:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
    if not answer_pattern:
        return []
    
    answer_str = answer_pattern.group(1).strip()
    answers_raw = re.split(r"[,\s/]+", answer_str)
    answers_raw = [x.upper() for x in answers_raw if x.strip()]

    option_map = {}
    for idx, child in enumerate(child_node):
        letter = chr(65 + idx)  # A, B, C...
        option_map[letter] = child
    
    chosen_nodes = []
    for ans in answers_raw:
        letter = ans.replace(".", "").strip()  
        if letter in option_map:
            chosen_nodes.append(letter)

    if not chosen_nodes:
        return []
    
    score_pattern = re.findall(r"([A-Z])\(\s*([\d\.]+)\s*\)", generated_text)
    score_dict = {k.upper(): float(v) for k, v in score_pattern}

    results = []
    for letter in chosen_nodes:
        c = score_dict.get(letter, 1.0)  
        results.append((option_map[letter], c))
    return results


def multi_choice_recursive_classify(
    requirement: str,
    merged_tree_file: str,
    sampled_data: List[Dict],
    sampled_embeddings: np.ndarray,
    max_depth: int = 5,
    top_k_per_level: int = 3,
) -> List[Dict]:

    from collections import deque

    start_path_info = {
        "path": "root",
        "confidence": 1.0,
        "history": [],
        "depth": 0
    }
    queue = deque([start_path_info])
    final_paths = []

    for depth in range(max_depth):
        if not queue:
            break
        next_level_queue = deque()

        while queue:
            path_info = queue.popleft()
            father_path = path_info["path"]
            father_conf = path_info["confidence"]
            father_history = path_info["history"]
            current_depth = path_info["depth"]

            child_nodes = get_child_node(merged_tree_file, father_path)
            if not child_nodes:
                final_paths.append(path_info)
                continue

            child_node_str = ""
            for idx, node in enumerate(child_nodes):
                child_node_str += f"{chr(65+idx)}. {node}\n"

            if current_depth == 0:
                user_prompt = (
                    f"Please follow the steps and choose correct answers from the following {len(child_nodes)} options as the requirements class according to requirement description.\n"
                    f"Step 1: Analyze the meaning of the requirements description provided.\n"
                    f"Step 2: Analyze the definition of each requirements class and check whether the given requirements description belongs to this class.\n"
                    f"Step 3: Choose correct answers of requirements class options that is suitable for the given requirement description.\n"
                    f"Step 4: Check whether your answer is right or not.\n"
                    f"###Requirement Description###\n{requirement}\n"
                    f"###Current Chosen Path###\n{father_path}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Please pick up to {top_k_per_level} sub-options, your answer format should be:\n"
                    f"Answer: A, C\nScores: A(0.8), C(0.6)\n"
                    f"###Answer###"
                )
            elif current_depth == 0:
                user_prompt = (
                    f"Please follow the steps and choose correct answers from the following {len(child_nodes)} options as the requirements class according to requirement description.\n"
                    f"Step 1: Analyze the meaning of the requirements description provided.\n"
                    f"Step 2: Analyze the definition of each requirements class and check whether the given requirements description belongs to this class.\n"
                    f"Step 3: Choose correct answers of requirements class options that is suitable for the given requirement description.\n"
                    f"Step 4: Check whether your answer is right or not.\n"
                    f"###Requirement Description###\n{requirement}\n"
                    f"###Current Chosen Path###\n{father_path}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Please pick up to {top_k_per_level} sub-options, your answer format should be:\n"
                    f"Answer: A, C\nScores: A(0.8), C(0.6)\n"
                    f"Only give the final answer, do not explain anything.\n"
                    f"###Answer###"
                    )
            else:
                user_prompt = (
                    f"Please continue to follow the steps above and choose correct answers from the following {len(child_nodes)} options as the requirements class according to requirement description and current chosen path.\n"
                    f"###Current Chosen Path###\n{father_path}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Pick up to {top_k_per_level} sub-options, your answer format should be:\n"
                    f"Answer: A, C\nScores: A(0.8), C(0.6)\n"
                    f"Only give the final answer, do not explain anything.\n"
                    f"###Answer###"
                )

            messages = [
                {"role": "system", "content": "You are an expert in requirements classification in the aerospace domain."}
            ]
            for (u_content, a_content) in father_history:
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": a_content})
            messages.append({"role": "user", "content": user_prompt})

            model_answer = call_model_api(messages)

            new_history = father_history + [(user_prompt, model_answer)]

            chosen_list = parse_multiple_answers(model_answer, child_nodes)
            if not chosen_list:
                final_paths.append({
                    "path": father_path,
                    "confidence": father_conf,
                    "history": new_history,
                    "depth": current_depth
                })
                continue

            chosen_list = sorted(chosen_list, key=lambda x: x[1], reverse=True)
            chosen_list = chosen_list[:top_k_per_level]

            for (child_name, child_conf) in chosen_list:
                new_path = father_path + "-" + child_name
                new_conf = father_conf * child_conf
                next_level_queue.append({
                    "path": new_path,
                    "confidence": new_conf,
                    "history": new_history,
                    "depth": current_depth + 1
                })

        queue = next_level_queue

    while queue:
        final_paths.append(queue.popleft())

    final_paths = sorted(final_paths, key=lambda x: x["confidence"], reverse=True)

    return final_paths[:3]


def batch_generate_answer_api(
    sentences: List[str],
    sampled_data: List[Dict],
    sampled_embeddings: np.ndarray,
    merged_tree: str,
    batch_answer: List[List[str]],
    max_depth: int = 5
) -> Tuple[List[List[str]], List[Dict], List[List[str]]]:

    final_results = []
    memory_total = []
    chat_history = []

    for s_idx, requirement_text in enumerate(sentences):
        s = requirement_text.strip()

        new_dict = {'Description': s}

        final_paths = multi_choice_recursive_classify(
            requirement=s,
            merged_tree_file=merged_tree,
            sampled_data = sampled_data,
            sampled_embeddings = sampled_embeddings,
            max_depth=max_depth,
            top_k_per_level=3,
        )
        top_3_paths_str = []
        for pinfo in final_paths:
            path_str = "-".join(pinfo["path"].split('-')[1:])
            conf = pinfo["confidence"]
            top_3_paths_str.append(f"{path_str} (conf={conf:.3f})")

        final_results.append(top_3_paths_str)
        
        if final_paths:
            best_path_info = final_paths[0]
            chat_history_thisround = best_path_info["history"]
        else:
            chat_history_thisround = []

        new_dict['Llm inference'] = top_3_paths_str 
        new_dict['Correct answer'] = batch_answer[s_idx] 
        memory_total.append(new_dict)
        chat_history.append(chat_history_thisround)

    
    return final_results, memory_total, chat_history

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

def count_processed_examples(answer_json_path):
    """统计已经写入answer.json的条数。"""
    count = 0
    try:
        with open(answer_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except FileNotFoundError:
        count = 0  
    return count

def load_memory_and_embedding(memory_path, embedding_path):
    memory_finally = []
    embedding_finally = []

    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    memory_finally.append(json.loads(line)['Example'])
    except FileNotFoundError:
        pass 

    try:
        embedding_finally = np.load(embedding_path).tolist()
    except FileNotFoundError:
        pass 

    return memory_finally, embedding_finally

def load_unprocessed_requirements(excel_path, processed_count):
    df = pd.read_excel(excel_path)
    df['Labels'] = df.apply(get_labels_from_row, axis=1)

    unprocessed_df = df.iloc[processed_count:].reset_index(drop=True)
    return unprocessed_df

def json_to_dataframe(json_file, processed_count=0):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []

    for item in data[processed_count:]:
        example = item.get('Example', {})
        description = example.get('Description', '').strip()
        labels = example.get('Correct answer', [])
        labels = [label for label in labels if label]  

        records.append({
            'Requirements Description': description,
            'Labels': labels
        })

    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    answer_json_path = 'Results/ToT0315/answer_taxonomy175_'+str(LLM_MODEL)+'.json'
    memory_json_path = 'Results/ToT0315/memory_taxonomy175_'+str(LLM_MODEL)+'.json'
    embedding_path = 'Results/ToT0315/embedding_taxonomy175_'+str(LLM_MODEL)+'.npy'

    json_path = 'selected_samples.json'
    merged_tree = 'final_structure_tree.json'
    sampled_data_path = 'leftover_samples.json'
    sampled_embeddings_path = 'leftover_samples_embeddings.npy'

    eval_batch_size = 1  

    processed_count = count_processed_examples(answer_json_path)

    memory_finally = []
    embedding_finally = []

    with open(sampled_data_path, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    sampled_embeddings = np.load(sampled_embeddings_path)
    
    unprocessed_data = json_to_dataframe(json_path, processed_count)
    
    
    for batch_start in tqdm(range(0, len(unprocessed_data), eval_batch_size), desc="Generating outputs"):
        batch = unprocessed_data['Requirements Description'][batch_start: batch_start + eval_batch_size].tolist()
        batch_answer = unprocessed_data['Labels'][batch_start: batch_start + eval_batch_size].tolist()

        responses, memory_total, chat_history = batch_generate_answer_api(
            batch,
            sampled_data,
            sampled_embeddings,
            merged_tree,
            batch_answer,
            max_depth=5
        )

        results = []
        current_memory = []
        current_embedding = []

        for example, top3_paths, memory, hist in zip(batch, responses, memory_total, chat_history):
            print("===")
            print(f"Final Top3 Output Paths: {top3_paths}\n")

            data_answer = {
                "conversations": [
                    {"from": "human", "type": "answer", "value": example},
                    {
                        "from": "gpt",
                        "value": top3_paths,
                        "history": [
                            {
                                "user_prompt": h[0],
                                "assistant_answer": h[1]
                            }
                            for h in hist
                        ]
                    }
                ]
            }
            results.append(data_answer)

            memory_finally.append(memory)
            current_memory.append(memory)

            emb = get_embedding_bge(example)
            embedding_finally.append(emb)
            current_embedding.append(emb)

        with open(answer_json_path, 'a', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        with open(memory_json_path, 'a', encoding='utf-8') as f:
            for entry in current_memory:
                data = {
                    "Example": {
                        "Description": entry['Description'],
                        "Llm inference": entry['Llm inference'],
                        "Correct answer": entry['Correct answer']
                    }
                }
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')

        np.save(embedding_path, np.array(embedding_finally))