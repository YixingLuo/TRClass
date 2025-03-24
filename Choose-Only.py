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

def get_all_paths(file_name, root):
    base_tree = load_tree_from_json(file_name)
    root_node = find_node_by_path(base_tree, [root])

    if root_node is None:
        return []

    all_paths = []

    def traverse(node, path):
        current_path = path + [node.get('name')]

        if not node.get('children'):
            all_paths.append('-'.join(current_path))
            return

        for child in node.get('children', []):
            traverse(child, current_path)

    traverse(root_node, [])
    return all_paths

def my_cosine_similarity(a,b):
    answer=0
    n=len(a)
    for i in range(n):
        answer+=a[i]*b[i]
    return answer

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device) 
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
    
    # Sort by similarity score in descending order and take top_k examples
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
        api_key="", ## 
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
            # If chunk.choices is empty, print usage
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
    if "none" in text_lower:
        return []

    letter_to_child = {}
    for idx, child in enumerate(child_node):
        letter_to_child[chr(ord('A') + idx)] = child

    score_pattern = re.findall(r"([A-Z])\(\s*([\d\.]+)\s*\)", generated_text)
    
    score_dict = {}
    letters_in_scores_order = []
    for letter, score_str in score_pattern:
        letter = letter.upper()
        if letter not in score_dict:
            letters_in_scores_order.append(letter)
        score_dict[letter] = float(score_str)

    if score_dict:
        results = []
        used_letters = set()
        for letter in letters_in_scores_order:
            if letter not in used_letters:  
                used_letters.add(letter)
                if letter in letter_to_child:
                    results.append((letter_to_child[letter], score_dict[letter]))
        return results

    chosen_letters: List[str] = []

    answer_match = re.search(r"answer:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
    if answer_match:
        answer_str = answer_match.group(1).strip()
        answers_raw = re.split(r"[,\s/;]+", answer_str)
        answers_raw = [ans.upper().replace(".", "") for ans in answers_raw if ans.strip()]
        chosen_letters.extend(answers_raw)
    else:
        lines = generated_text.splitlines()
        for line in lines:
            line_strip = line.strip()
            if "," in line_strip:
                parts = [x.strip().upper().replace(".", "") for x in line_strip.split(",")]
                if all(re.match(r"^[A-Z]$", p) for p in parts):
                    chosen_letters.extend(parts)
                    continue  

            match = re.match(r"^([A-Z])(?:[\.\uff0e\s]|$)", line_strip)
            if match:
                chosen_letters.append(match.group(1).upper())

    results: List[Tuple[str, float]] = []
    used_letters = set()
    for letter in chosen_letters:
        if letter not in used_letters:
            used_letters.add(letter)
            if letter in letter_to_child:
                results.append((letter_to_child[letter], 1.0))

    return results

def one_shot_classify(
    requirement: str,
    merged_tree_file: str,
    top_k: int = 3
) -> Tuple[List[Tuple[str, float]], str, str]:


    all_leaf_paths = get_all_paths(merged_tree_file, "root")
    all_leaf_paths = [p.replace("root-", "") for p in all_leaf_paths]

    child_node_str = ""
    for idx, path in enumerate(all_leaf_paths):
        letter = chr(65 + idx)
        child_node_str += f"{letter}. {path}\n"


    user_prompt = (
                    f"Please follow the steps and choose correct answers from the following {len(child_node_str)} options as the requirements class according to requirement description.\n"
                    f"###Requirement Description###\n{requirement}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Please pick up to {top_k} sub-options, your answer format MUST be like this:\n"
                    f"Answer: A, C, D\n"
                    f"Scores: A(0.9), C(0.7), D(0.5)\n"
                    f"###Answer###"
                )

    messages = [
        {"role": "system", "content": "You are an expert in requirements classification in the aerospace domain."},
        {"role": "user", "content": user_prompt},
    ]

    model_answer = call_model_api(messages)

    chosen_list = parse_multiple_answers(model_answer, all_leaf_paths)

    chosen_list = sorted(chosen_list, key=lambda x: x[1], reverse=True)[:top_k]
     
    return chosen_list, user_prompt, model_answer


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
        chosen_list, user_prompt, model_answer = one_shot_classify(
            requirement=s,
            merged_tree_file=merged_tree,
            top_k=3
        )
        top_3_paths_str = []
        for path_str, conf in chosen_list:
            top_3_paths_str.append(f"{path_str} (conf={conf:.3f})")

        final_results.append(top_3_paths_str)

        this_history = [
            ("User Prompt", user_prompt),
            ("Assistant Answer", model_answer)
        ]
        chat_history.append(this_history)

        new_dict['Llm inference'] = top_3_paths_str
        new_dict['Correct answer'] = batch_answer[s_idx]
        memory_total.append(new_dict)

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
                    memory_finally.append(json.loads(line)['example'])
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
        labels = [label for label in labels if label]  # 去除空标签

        records.append({
            'Requirements Description': description,
            'Labels': labels
        })

    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    answer_json_path = 'Results/answer_raw175_'+str(LLM_MODEL)+'.json'
    memory_json_path = 'Results/memory_raw175_'+str(LLM_MODEL)+'.json'
    embedding_path = 'Results/embedding_raw175_'+str(LLM_MODEL)+'.npy'

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
