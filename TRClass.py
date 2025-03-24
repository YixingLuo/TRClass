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
    return data # 假设 JSON 文件的顶层是一个带有 "root" 键的字典

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
    """
    father 形如 "root-功能需求-数据需求"，函数会找到 father 最末尾节点的子节点列表。
    """
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

tokenizer = AutoTokenizer.from_pretrained('D:/LLM-code/LLM_Models/bert-base-chinese')
model = AutoModel.from_pretrained('D:/LLM-code/LLM_Models/bert-base-chinese')
# tokenizer = AutoTokenizer.from_pretrained('D:/LLM-code/LLM_Models/bge-base-zh-v1.5')
# model = AutoModel.from_pretrained('D:/LLM-code/LLM_Models/bge-base-zh-v1.5')
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
    """
    对中英文混合句子进行分词
    """
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

     # 计算 embedding_finally 每个向量与 embedding 的点积
    dot_products = np.dot(embedding_finally, embedding)
    
    # 计算 embedding_finally 中每个向量的范数
    norms_finally = np.linalg.norm(embedding_finally, axis=1)
    
    # 计算 embedding 的范数
    norm_embedding = np.linalg.norm(embedding)
    
    # 计算余弦相似度：点积除以两个向量的范数的乘积
    cosine_similarities = dot_products / (norms_finally * norm_embedding)
    
    # 获取余弦相似度最高的三个向量的下标
    top3_indices = np.argsort(cosine_similarities)[::-1][:3]
    return top3_indices


def get_example(memory_finally,s, embedding_finally, merged_tree):
    if not memory_finally:
        return ""
    embedding = get_embedding_bge(s)
    # memory_finally2=[item['Description'] for item in memory_finally]
    # rouge_L, index_list = compute_rouge_multiple3(memory_finally2, s)

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
        # answer += "Requirement Description:"+memory_finally[max_index]["Description"]+final_ans 
        answer += "Requirement Description:"+memory_finally[max_index]["Description"]+ "\\n Answer: " + memory_finally[max_index]["Correct answer"][0]+ "\\n"
        # + "\\nExplaination: "+memory_finally[max_index]["summary"] + "\\n"

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
    
    # rouge_L, index_list = compute_rouge_multiple3(sampled_data, s)
    # example_texts = []
    # for match in index_list:
    #     example_texts.append(
    #         f"Requirement Description: {sampled_data[match]['Example']['Description']}\n"
    #         f"Answer: {','.join([ans for ans in sampled_data[match]['Example']['Correct answer'] if ans])}\n"
    #     )

    example_texts = []
    for match in top_matches:
        example_texts.append(
            f"Requirement Description: {match[0]['Example']['Description']}\n"
            f"Answer: {','.join([ans for ans in match[0]['Example']['Correct answer'] if ans])}\n"
        )
    
    return "\n".join(example_texts)

def extract_json(response_content):
    # 匹配 ```json 包裹的代码块
    json_block_pattern = r"```json\s*(\{.*?\}|\[.*?\])\s*```"
    matches = re.findall(json_block_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    # 如果没有匹配到代码块，尝试直接匹配裸 JSON
    json_pattern = r"(\{.*?\}|\[.*?\])"
    matches = re.findall(json_pattern, response_content, re.DOTALL)
    if matches:
        for match in matches:
            try:
                # 尝试解析为 JSON
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # 如果还是没有匹配到有效 JSON，返回None
    return None

# LLM_MODEL = "deepseek-chat"
# LLM_MODEL = "deepseek-reasoner"
# LLM_MODEL = "qwen2.5-32b-instruct"
LLM_MODEL = "qwen2.5-7b-instruct"
# LLM_MODEL = "qwq-32b"
# LLM_MODEL = "gpt-3.5-turbo"
def call_model_api(messages: List[Dict[str, str]]):
    client = OpenAI(
        # api_key = "sk-870beb4daf0542938e2877fd88d77c45",
        # base_url = "https://api.deepseek.com" ## deepseek
        # api_key = "bS3NzfR0AUYYsao6MUSLUigW8l6EJgsw", ## deepseek-distill
        # base_url = "https://api.deepinfra.com/v1/openai",
        api_key="sk-b8bef0abe2524b0fbc5fbaafd1faec8d", ## qwen
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        # api_key='sk-bO2s4M7N4jy8CQzreJNPPgT3Lt4pa1CfSXVH4eEdR7qdeiNi',
        # base_url='https://api.openai-proxy.org/v1',
        # base_url="https://api2.aigcbest.top/v1",
        # api_key="sk-nobdJGUuBgh6VG5lgWWznMNhVusWvqh2XU0IdAdhvLMInMb6",
    )
    if LLM_MODEL != "qwq-32b":
        response = client.chat.completions.create(
            # model="qwen2.5-32b-instruct",
            model=LLM_MODEL,
            # model = "deepseek-reasoner",
            # model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            messages=messages,
            temperature=0,
            timeout=60
        )
        if '</think>' in response.choices[0].message.content:
            response_content = response.choices[0].message.content.split('</think>')[-1]
            print(f'llm输出：{response_content}')
            return response_content
        else:
            response_content = response.choices[0].message.content
            print(f'llm输出：{response_content}')
            return response_content
    else:
        response = client.chat.completions.create(
                # model="qwen2.5-32b-instruct",
                model=LLM_MODEL,
                # model = "deepseek-reasoner",
                # model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
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

# ===============  解析多选答案  ===============
def parse_multiple_answers(generated_text: str, child_node: List[str]) -> List[Tuple[str, float]]:
    """
    从模型返回的文本中解析多选答案和可选的置信度分值（如果有）。
    返回一个列表，元素是 (child_name, confidence)，confidence 没有解析到就默认 1.0。
    如果答案里出现 "none" 则返回空列表。
    
    示例:
    generated_text = 
        \"Answer: A, C
         Scores: A(0.9), C(0.7)\"
    child_node = [子节点1, 子节点2, 子节点3, ...]
    -> [("子节点1", 0.9), ("子节点3", 0.7)] 
    """
    text_lower = generated_text.lower()
    # if "none" in text_lower:
    #     return []
    
    # 在 "answer:" 后的内容中找选项
    # 可以用正则或简单切分，假定格式大概是 "Answer: A, C"
    answer_pattern = re.search(r"answer:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
    if not answer_pattern:
        return []
    
    answer_str = answer_pattern.group(1).strip()
    # 可能出现 'A, C' 或 'A B' 等
    # 用逗号、空格、分号等切分
    # 同时只保留在 0~len(child_node)-1 范围内的选项
    # 例如: child_node = ["登陆", "注册", "搜索"] => A->登陆, B->注册, C->搜索
    answers_raw = re.split(r"[,\s/]+", answer_str)
    answers_raw = [x.upper() for x in answers_raw if x.strip()]

    # 构建映射 A->0, B->1, C->2, ...
    option_map = {}
    for idx, child in enumerate(child_node):
        letter = chr(65 + idx)  # A, B, C...
        option_map[letter] = child
    
    chosen_nodes = []
    for ans in answers_raw:
        letter = ans.replace(".", "").strip()  # 去掉可能的 "A." 
        if letter in option_map:
            chosen_nodes.append(letter)

    if not chosen_nodes:
        return []
    
    # 如果有 "Scores:" 这类信息, 我们用正则把分值解析出来
    # 格式示例: "Scores: A(0.9), C(0.7)" 
    # 当然如果没有这种 scores 信息，就默认 1.0
    score_pattern = re.findall(r"([A-Z])\(\s*([\d\.]+)\s*\)", generated_text)
    score_dict = {k.upper(): float(v) for k, v in score_pattern}

    results = []
    for letter in chosen_nodes:
        c = score_dict.get(letter, 1.0)  # 没有解析到就默认 1.0
        results.append((option_map[letter], c))
    return results


# ===============  多选递归分类函数  ===============
def multi_choice_recursive_classify(
    requirement: str,
    merged_tree_file: str,
    # memory_finally: List[Dict],
    # embedding_finally: List,
    sampled_data: List[Dict],
    sampled_embeddings: np.ndarray,
    max_depth: int = 5,
    top_k_per_level: int = 3,
) -> List[Dict]:
    """
    对单条需求 requirement 进行多选 BFS 分类。
    - 第 0 层时，把根据 memory 生成的示例一起拼到 prompt 中。
    - 返回按置信度排序的前 3 条路径(可自行改)。
    """
    from collections import deque

    start_path_info = {
        "path": "root",
        "confidence": 1.0,
        "history": [],
        "depth": 0
    }
    queue = deque([start_path_info])
    final_paths = []

    # 先获取 example 文本(只在第 0 层用到)
    # example_text = get_example(memory_finally, requirement, embedding_finally, merged_tree_file)
    example_text = get_example_from_sampled_data(sampled_data, sampled_embeddings, requirement, top_k=3)

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
                # 到叶子
                final_paths.append(path_info)
                continue

            # 组装选项
            child_node_str = ""
            for idx, node in enumerate(child_nodes):
                child_node_str += f"{chr(65+idx)}. {node}\n"

            # === 构造 Prompt ===
            # 如果在第 0 层，则拼接示例
            # father_path = "-".join(father_path.split('-')[1:])
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
                    # f"If none of the above options are correct, your answer should be: None.\n"
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
                    # f"If none of the above options are correct, your answer should be: None.\n"
                    f"Only give the final answer, do not explain anything.\n"
                    f"###Answer###"
                    )
            else:
                user_prompt = (
                    # f"###Requirement Description###\n{requirement}\n"
                    f"Please continue to follow the steps above and choose correct answers from the following {len(child_nodes)} options as the requirements class according to requirement description and current chosen path.\n"
                    f"###Current Chosen Path###\n{father_path}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Pick up to {top_k_per_level} sub-options, your answer format should be:\n"
                    f"Answer: A, C\nScores: A(0.8), C(0.6)\n"
                    # f"If none of the above options are correct, your answer should be: None.\n"
                    f"Only give the final answer, do not explain anything.\n"
                    f"###Answer###"
                )

            # 合成 messages
            messages = [
                {"role": "system", "content": "You are an expert in requirements classification in the aerospace domain."}
            ]
            for (u_content, a_content) in father_history:
                messages.append({"role": "user", "content": u_content})
                messages.append({"role": "assistant", "content": a_content})
            messages.append({"role": "user", "content": user_prompt})

            # 调用模型
            model_answer = call_model_api(messages)
            # 更新 history
            new_history = father_history + [(user_prompt, model_answer)]

            # 解析多选
            chosen_list = parse_multiple_answers(model_answer, child_nodes)
            if not chosen_list:
                final_paths.append({
                    "path": father_path,
                    "confidence": father_conf,
                    "history": new_history,
                    "depth": current_depth
                })
                continue

            # 选置信度最高的 top_k_per_level
            chosen_list = sorted(chosen_list, key=lambda x: x[1], reverse=True)
            chosen_list = chosen_list[:top_k_per_level]

            # 依次扩展
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

    # 把 BFS 里剩余的路径也放到 final
    while queue:
        final_paths.append(queue.popleft())

    # 排序选置信度最高的
    final_paths = sorted(final_paths, key=lambda x: x["confidence"], reverse=True)
    # 截取前三
    return final_paths[:3]


def batch_generate_answer_api(
    sentences: List[str],
    # memory_finally: List[Dict],
    # embedding_finally: List,
    sampled_data: List[Dict],
    sampled_embeddings: np.ndarray,
    merged_tree: str,
    batch_answer: List[List[str]],
    max_depth: int = 5
) -> Tuple[List[List[str]], List[Dict], List[List[str]]]:
    """
    针对一个 batch 的需求描述 sentences（每个需求与对应的答案 batch_answer），
    进行多选递归分类，并返回：
     - final_results: 每条需求对应的【前3条路径字符串】列表
     - memory_total: 用于更新 memory 的信息
     - chat_history: 每条需求的对话历史（示例简化处理，仅存最优路径的对话）
    """
    final_results = []
    memory_total = []
    chat_history = []

    for s_idx, requirement_text in enumerate(sentences):
        s = requirement_text.strip()
        # 构建要写到 memory 里的新字典
        new_dict = {'Description': s}

        # print('开启新的推理:', s)

        # 执行多选递归分类
        final_paths = multi_choice_recursive_classify(
            requirement=s,
            merged_tree_file=merged_tree,
            # memory_finally=memory_finally,
            # embedding_finally=embedding_finally,
            sampled_data = sampled_data,
            sampled_embeddings = sampled_embeddings,
            max_depth=max_depth,
            top_k_per_level=3,
        )

        # final_paths 里每个元素都有 path / confidence / history
        # path 形如 "root-功能需求-性能需求" 等
        # 这里我们只取 path(去掉 root) 形成给定的结果
        # 以及把前3条都加进来
        top_3_paths_str = []
        for pinfo in final_paths:
            path_str = "-".join(pinfo["path"].split('-')[1:])
            conf = pinfo["confidence"]
            top_3_paths_str.append(f"{path_str} (conf={conf:.3f})")
        print("本需求最终 top3 路径：", top_3_paths_str)

        # 将 top_3_paths_str 整理输出
        final_results.append(top_3_paths_str)
        
        # 这里假设只保存第一条路径对应的对话历史
        # 也可以都保存，自己根据需要
        if final_paths:
            best_path_info = final_paths[0]
            chat_history_thisround = best_path_info["history"]
        else:
            chat_history_thisround = []

        new_dict['Llm inference'] = top_3_paths_str  # 这里存全部 top3
        new_dict['Correct answer'] = batch_answer[s_idx]  # 真实标签(列表)
        memory_total.append(new_dict)
        chat_history.append(chat_history_thisround)

        # 更新 memory & embedding
        # 如果需要将本条需求加入 memory，就把它加上
        # 以及获取 embedding
        # （视具体需求而定）
    
    return final_results, memory_total, chat_history

def get_labels_from_row(row):
    labels = []
    for i in range(1, 3):  # 支持两条路径
    # for i in range(1, 2):  # 支持两条路径    
        level_path = []
        for j in range(1, 5):  # 每条路径最多4级
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
        count = 0  # 如果文件不存在，说明是第一次运行
    return count

def load_memory_and_embedding(memory_path, embedding_path):
    """读取已有的memory和embedding."""
    memory_finally = []
    embedding_finally = []

    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    memory_finally.append(json.loads(line)['Example'])
    except FileNotFoundError:
        pass  # memory文件不存在，说明是首次运行

    try:
        embedding_finally = np.load(embedding_path).tolist()
    except FileNotFoundError:
        pass  # embedding文件不存在，说明是首次运行

    return memory_finally, embedding_finally

def load_unprocessed_requirements(excel_path, processed_count):
    """从Excel读取未处理的requirements。"""
    df = pd.read_excel(excel_path)
    df['Labels'] = df.apply(get_labels_from_row, axis=1)

    # 跳过已处理的部分
    unprocessed_df = df.iloc[processed_count:].reset_index(drop=True)
    return unprocessed_df

def json_to_dataframe(json_file, processed_count=0):
    """
    从JSON文件中读取数据，并转换为DataFrame格式，从processed_count之后的记录开始读取。

    参数:
        json_file (str): JSON文件路径。
        processed_count (int): 已处理过的example数量，默认从第0个开始。

    返回:
        DataFrame: 包含'Requirements Description' 和 'Labels' 两个字段。
    """
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


# ===============  主函数示例  ===============
if __name__ == '__main__':
    answer_json_path = 'Results/ToT0315/answer_taxonomyshot175_'+str(LLM_MODEL)+'.json'
    memory_json_path = 'Results/ToT0315/memory_taxonomyshot175_'+str(LLM_MODEL)+'.json'
    embedding_path = 'Results/ToT0315/embedding_taxonomyshot175_'+str(LLM_MODEL)+'.npy'

    # excel_path = 'D:/LLM-code/ReqMulti/原始数据/revised_new_taxonomy_shuffle.xlsx'
    json_path = 'D:/LLM-code/ReqMulti/原始数据/175_selected_samples.json'
    merged_tree = 'D:/LLM-code/ReqMulti/原始数据/final_structure_tree.json'
    sampled_data_path = 'D:/LLM-code/ReqMulti/原始数据/175_leftover_samples.json'
    sampled_embeddings_path = 'D:/LLM-code/ReqMulti/原始数据/175_leftover_samples_embeddings.npy'

    eval_batch_size = 1  # 每批大小

    # 获取已处理数量
    processed_count = count_processed_examples(answer_json_path)
    print(f'已经处理 {processed_count} 条')

    # 读取历史memory和embedding
    # memory_finally, embedding_finally = load_memory_and_embedding(memory_json_path, embedding_path)
    memory_finally = []
    embedding_finally = []
    # print(f'历史memory加载 {len(memory_finally)} 条')
    # print(f'历史embedding加载 {len(embedding_finally)} 条')
    with open(sampled_data_path, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    sampled_embeddings = np.load(sampled_embeddings_path)
    print(f'samples加载 {len(sampled_data)} 条')
    print(f'embedding加载 {len(sampled_embeddings)} 条')
    

    # 读取未处理的requirements
    # unprocessed_data = load_unprocessed_requirements(excel_path, processed_count)
    unprocessed_data = json_to_dataframe(json_path, processed_count)
    
    
    for batch_start in tqdm(range(0, len(unprocessed_data), eval_batch_size), desc="Generating outputs"):
        batch = unprocessed_data['Requirements Description'][batch_start: batch_start + eval_batch_size].tolist()
        batch_answer = unprocessed_data['Labels'][batch_start: batch_start + eval_batch_size].tolist()

        # 这里 batch_answer 是个列表，其中每个元素也是列表
        # 如果原先只存了一个路径，就类似 [["功能需求-数据需求"], ["接口需求-..."], ...]
        responses, memory_total, chat_history = batch_generate_answer_api(
            batch,
            sampled_data,
            sampled_embeddings,
            merged_tree,
            batch_answer,
            max_depth=5
        )

        # 将结果写入
        results = []
        current_memory = []
        current_embedding = []

        for example, top3_paths, memory, hist in zip(batch, responses, memory_total, chat_history):
            print("===")
            # print(f"Final Input: {example}")
            print(f"Final Top3 Output Paths: {top3_paths}\n")

            # 写 answer.json
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

            # 更新 memory
            memory_finally.append(memory)
            current_memory.append(memory)

            # 更新 embedding
            emb = get_embedding_bge(example)
            embedding_finally.append(emb)
            current_embedding.append(emb)

        # 保存到 answer.json
        with open(answer_json_path, 'a', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

        # 保存 memory.json
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

        # 保存 embedding.npy
        np.save(embedding_path, np.array(embedding_finally))