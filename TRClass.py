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
import faiss  # 添加Faiss库

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

def my_cosine_similarity(a, b):
    answer = 0
    n = len(a)
    for i in range(n):
        answer += a[i] * b[i]
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

def get_example(memory_finally, s, faiss_index, merged_tree):
    if not memory_finally:
        return ""
    
    # 使用Faiss进行检索
    embedding = get_embedding_bge(s).astype('float32').reshape(1, -1)
    _, index_list = faiss_index.search(embedding, 3)  # 获取前3个最相似的索引
    index_list = index_list[0]  # 展平结果

    answer = ""
    for i in range(len(index_list)):
        answer += "Example " + str(i) + "\n"
        max_index = index_list[i]
        father = "root"
        ans_list = memory_finally[max_index]["Correct answer"][0].split('-')
        final_ans = ""
        for item in ans_list:
            child_node = get_child_node(merged_tree, father)
            child_node_str = ""
            for idx in range(len(child_node)):
                select = chr(65 + idx) + ". "  
                child_node_str += select + child_node[idx] + "\n"
            final_ans += "\nOptions: \n"
            final_ans += child_node_str
            for idx in range(len(child_node)):
                if child_node[idx] == item:
                    final_ans += "Answer: " + chr(65 + idx) + ". " + item
                    break
            father = father + "-" + item

        answer += "Requirement Description:" + memory_finally[max_index]["Description"] + "\n Answer: " + memory_finally[max_index]["Correct answer"][0] + "\n"

    return answer

def get_example_from_sampled_data(sampled_data, faiss_index, s, top_k=3):
    """Retrieve the top_k most similar examples from sampled_data.json using Faiss."""
    if not sampled_data:
        return ""
    
    embedding = get_embedding_bge(s).astype('float32').reshape(1, -1)
    _, indices = faiss_index.search(embedding, top_k)
    indices = indices[0]  # 展平结果
    
    example_texts = []
    for idx in indices:
        entry = sampled_data[idx]
        example_texts.append(
            f"Requirement Description: {entry['Example']['Description']}\n"
            f"Answer: {','.join([ans for ans in entry['Example']['Correct answer'] if ans])}\n"
        )
    
    return "\n".join(example_texts)

def get_example_from_sampled_data_bm25_knn_rerank(sampled_data, faiss_index, requirement_text, top_k_bm25=20, top_k_knn=20, top_k_final=3):
    # BM25 检索
    corpus = [entry["Example"]["Description"] for entry in sampled_data]
    tokenized_corpus = [tokenize_chinese_english(doc).split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query = tokenize_chinese_english(requirement_text).split()
    bm25_scores = bm25.get_scores(query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

    # Faiss向量检索
    query_vec = get_embedding_bge(requirement_text).astype('float32').reshape(1, -1)
    _, knn_top_indices = faiss_index.search(query_vec, top_k_knn)
    knn_top_indices = knn_top_indices[0]  # 展平结果

    # 合并两种检索结果
    merged_indices = list(set(bm25_top_indices).union(set(knn_top_indices)))

    # ===== BGE-Reranker rerank =====
    rerank_pairs = [(requirement_text, sampled_data[i]["Example"]["Description"]) for i in merged_indices]
    texts = reranker_tokenizer.batch_encode_plus(rerank_pairs, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        scores = reranker_model(**texts).logits.squeeze(-1).cpu().numpy()

    reranked = sorted(zip(merged_indices, scores), key=lambda x: x[1], reverse=True)[:top_k_final]

    # ===== 构造最终结果 =====
    example_texts = []
    for idx, score in reranked:
        entry = sampled_data[idx]["Example"]
        desc = entry.get("Description", "")
        answers = entry.get("Correct answer", [])
        example_texts.append(
            f"Requirement Description: {desc}\n"
            f"Answer: {', '.join([ans for ans in answers if ans])}\n"
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

LLM_MODEL = "qwen2.5-7b-instruct"

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
            print(f'{response_content}')
            return response_content
        else:
            response_content = response.choices[0].message.content
            print(f'{response_content}')
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
                # Print reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    if delta.content != "" and is_answering is False:
                        print("\n" + "=" * 20 + "content" + "=" * 20 + "\n")
                        is_answering = True
                    # Print content
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
    faiss_index,
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

    example_text = get_example_from_sampled_data_bm25_knn_rerank(sampled_data, faiss_index, requirement)

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
                    f"Below are some similar requirement examples:\n"
                    f"{example_text}\n\n"
                    f"Step 1: Analyze the meaning of the requirements description provided.\n"
                    f"Step 2: Analyze the definition of each requirements class and check whether the given requirements description belongs to this class.\n"
                    f"Step 3: Choose correct answers of requirements class options that is suitable for the given requirement description.\n"
                    f"Step 4: Check whether your answer is right or not.\n"
                    f"Step 5: Take the similar requirements description shown in the example to answer this question.\n"
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
    faiss_index,
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
            sampled_data=sampled_data,
            faiss_index=faiss_index,
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
        embedding_finally = np.load(embedding_path)
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
    answer_json_path = 'Results/'+str(LLM_MODEL)+'.json'
    memory_json_path = 'Results/'+str(LLM_MODEL)+'.json'
    embedding_path = 'Results/'+str(LLM_MODEL)+'.npy'

    json_path = 'selected_samples.json'
    merged_tree = 'final_structure_tree.json'
    sampled_data_path = 'leftover_samples.json'
    sampled_embeddings_path = 'leftover_samples_embeddings.npy'

    eval_batch_size = 1  

    processed_count = count_processed_examples(answer_json_path)

    # 加载数据和创建Faiss索引
    memory_finally, embedding_finally_np = load_memory_and_embedding(memory_json_path, embedding_path)
    
    # 创建memory_finally的Faiss索引
    memory_faiss_index = None
    if len(memory_finally) > 0:
        embedding_finally_np = embedding_finally_np.astype('float32')
        dimension = embedding_finally_np.shape[1]
        memory_faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积(IP)作为相似度度量
        memory_faiss_index.add(embedding_finally_np)
    
    # 加载sampled_data和创建其Faiss索引
    with open(sampled_data_path, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    sampled_embeddings = np.load(sampled_embeddings_path).astype('float32')
    
    dimension = sampled_embeddings.shape[1]
    sampled_faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积(IP)作为相似度度量
    sampled_faiss_index.add(sampled_embeddings)
    
    unprocessed_data = json_to_dataframe(json_path, processed_count)
    
    for batch_start in tqdm(range(0, len(unprocessed_data), eval_batch_size), desc="Generating outputs"):
        batch = unprocessed_data['Requirements Description'][batch_start: batch_start + eval_batch_size].tolist()
        batch_answer = unprocessed_data['Labels'][batch_start: batch_start + eval_batch_size].tolist()

        responses, memory_total, chat_history = batch_generate_answer_api(
            batch,
            sampled_data,
            sampled_faiss_index,
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
            
            # 更新embedding_finally和Faiss索引
            if memory_faiss_index is None:
                # 第一次创建索引
                dimension = len(emb)
                memory_faiss_index = faiss.IndexFlatIP(dimension)
                memory_faiss_index.add(emb.reshape(1, -1).astype('float32'))
                embedding_finally_np = emb.reshape(1, -1).astype('float32')
            else:
                # 添加到现有索引
                memory_faiss_index.add(emb.reshape(1, -1).astype('float32'))
                embedding_finally_np = np.vstack([embedding_finally_np, emb.reshape(1, -1).astype('float32')])
            
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

        np.save(embedding_path, embedding_finally_np)
