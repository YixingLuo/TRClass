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

def get_all_paths(file_name, root):
    """
    输入根节点名称，返回从根节点到所有叶子节点的全部路径。
    每个路径表示为以'-'连接的字符串。
    """
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
        # father = "root"
        # ans_list = memory_finally[max_index]["Correct answer"][0].split('-')
        # final_ans = ""
        # for item in ans_list:
        #     child_node = get_child_node(merged_tree, father)
        #     child_node_str = ""
        #     for idx in range(len(child_node)):
        #         select = chr(65 + idx) + ". "  
        #         child_node_str += select + child_node[idx] + "\\n"
        #     final_ans += "\\nOptions: \\n"
        #     final_ans += child_node_str
        #     for idx in range(len(child_node)):
        #         if child_node[idx] == item:
        #             final_ans += "Answer: " + chr(65 + idx) + ". " + item
        #             break
        #     father = father + "-" + item
        # answer += "Requirement Description:"+memory_finally[max_index]["Description"]+final_ans 
        answer += "Requirement Description:"+memory_finally[max_index]["Description"]+ "\\n Answer: " + ','.join(memory_finally[max_index]["Correct answer"])+ "\\n"
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
LLM_MODEL = "qwq-32b"
# LLM_MODEL = "gpt-3.5-turbo"
# LLM_MODEL = "Qwen/QwQ-32B"
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
            # model = "Qwen/QwQ-32B",
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
                # model = "Qwen/QwQ-32B",
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
    返回一个列表，元素是 (child_name, confidence)，
    若没有解析到confidence则默认1.0。
    
    若文本包含"none"(不区分大小写)，则返回空列表。
    
    优先级：
    1) 若在文本中找到了类似 "A(0.9), B(0.8)" 这样的评分信息，则它决定了最终选项以及分值。
       - 即只取出出现在“评分”里的字母作为答案，分值从括号内解析。
    2) 否则，如果文本中存在 "Answer:" 则解析 "Answer:" 后的部分作为选项。
    3) 若也没有，则再尝试从整行逗号分隔的字母中提取 (如 "B, O, K")，或从行首的单个大写字母中提取 (如 "D. ...行")。
    4) 所有从2、3步获取的字母，若没有匹配到评分信息（因为可能只写了部分），则其置信度默认1.0。

    示例：
        "B, O, J
         ###Scores###
         B(0.9), O(0.8), J(0.7)"
    或：
        "Answer: A, C
         Scores: A(0.9), C(0.7)"
    或：
        "B, O, K
         Scores: B(0.9), O(0.7), K(0.5)"
    或：
        "P
         ###Scores###
         P(1.0)"
    """

    text_lower = generated_text.lower()
    # (1) 若包含“none”，直接返回空
    if "none" in text_lower:
        return []

    # (2) 建立 A->child_node[0], B->child_node[1], ...
    letter_to_child = {}
    for idx, child in enumerate(child_node):
        letter_to_child[chr(ord('A') + idx)] = child

    # (3) 先全局查找形如 "X(0.9)" 的评分信息
    #     若有，则优先使用它们来决定答案
    score_pattern = re.findall(r"([A-Z])\(\s*([\d\.]+)\s*\)", generated_text)
    
    # 我们用一个列表来记录出现顺序，再用字典来存分数
    # 若同一个字母重复出现，则最后一次出现覆盖前面的
    score_dict = {}
    letters_in_scores_order = []
    for letter, score_str in score_pattern:
        letter = letter.upper()
        if letter not in score_dict:
            letters_in_scores_order.append(letter)
        score_dict[letter] = float(score_str)

    # (4) 若找到了任何评分信息，就直接根据评分信息构造结果
    if score_dict:
        results = []
        used_letters = set()
        for letter in letters_in_scores_order:
            if letter not in used_letters:  # 避免重复
                used_letters.add(letter)
                # 映射到 child_node
                if letter in letter_to_child:
                    results.append((letter_to_child[letter], score_dict[letter]))
        return results

    # (5) 若没有评分信息，则解析其他答案形式
    chosen_letters: List[str] = []

    # (5.1) 优先检查 "Answer:"
    answer_match = re.search(r"answer:\s*([^\n\r]+)", generated_text, re.IGNORECASE)
    if answer_match:
        # 例如 "Answer: A, C"
        answer_str = answer_match.group(1).strip()
        # 用逗号、空格、分号、斜杠等切分
        answers_raw = re.split(r"[,\s/;]+", answer_str)
        # 去掉可能的句点、大写化
        answers_raw = [ans.upper().replace(".", "") for ans in answers_raw if ans.strip()]
        chosen_letters.extend(answers_raw)
    else:
        # (5.2) 如果没有 "Answer:"，尝试从每行解析
        lines = generated_text.splitlines()
        for line in lines:
            line_strip = line.strip()
            # 先看看是否整行形如 "B, O, K"
            #   ^[A-Za-z]([\s]*,[\s]*[A-Za-z])+
            #   但只想要大写字母 => [A-Z]
            # 这里用个简单判断：如果行里包含逗号，就按逗号切分，看看是不是都是大写字母
            if "," in line_strip:
                parts = [x.strip().upper().replace(".", "") for x in line_strip.split(",")]
                # 检查 parts 是否每个都是单个大写字母
                if all(re.match(r"^[A-Z]$", p) for p in parts):
                    chosen_letters.extend(parts)
                    continue  # 这一行处理完，就不再匹配行首字母了
            
            # 如果不是逗号形式，就匹配行首单字母 + 可选句点
            match = re.match(r"^([A-Z])(?:[\.\uff0e\s]|$)", line_strip)
            if match:
                chosen_letters.append(match.group(1).upper())

    # (6) 去重 + 映射到 child_node + 默认置信度 1.0
    results: List[Tuple[str, float]] = []
    used_letters = set()
    for letter in chosen_letters:
        if letter not in used_letters:
            used_letters.add(letter)
            if letter in letter_to_child:
                # 没有评分信息 => 默认1.0
                results.append((letter_to_child[letter], 1.0))

    return results


# ===============  多选递归分类函数  ===============
def one_shot_classify(
    requirement: str,
    merged_tree_file: str,
    sampled_data: List[Dict],
    sampled_embeddings: List,
    top_k: int = 3
) -> Tuple[List[Tuple[str, float]], str, str]:
    """
    一次性将所有“叶子路径”作为分类选项，让大模型选择最匹配的top_k。
    返回：
      - chosen_list: [(path_str, confidence), ...] 排序后的top_k
      - user_prompt: 方便后续存history
      - model_answer: 大模型原文
    """

    all_leaf_paths = get_all_paths(merged_tree_file, "root")
    # 如果想去掉 "root-"，可以自行处理
    all_leaf_paths = [p.replace("root-", "") for p in all_leaf_paths]

    # 2. 构造选项字符串
    child_node_str = ""
    for idx, path in enumerate(all_leaf_paths):
        letter = chr(65 + idx)
        child_node_str += f"{letter}. {path}\n"

    # 3. （可选）检索相似示例
    # example_text = get_example_from_sampled_data(sampled_data, sampled_embeddings, requirement, top_k=2)

    # 4. 拼接 Prompt
    # 这里示例给出一个通用形式，你可根据需要改动

    # 先获取 example 文本(只在第 0 层用到)
    # example_text = get_example(memory_finally, requirement, embedding_finally, merged_tree_file)
    example_text = get_example_from_sampled_data(sampled_data, sampled_embeddings, requirement, top_k=3)

    user_prompt = (
                    f"Please follow the steps and choose correct answers from the following {len(child_node_str)} options as the requirements class according to requirement description.\n"
                    f"Below are some similar requirement examples:\n"
                    f"{example_text}\n\n"
                    # f"Step 1: Analyze the meaning of the requirements description provided.\n"
                    # f"Step 2: Analyze the definition of each requirements class and check whether the given requirements description belongs to this class.\n"
                    # f"Step 3: Choose correct answers of requirements class options that is suitable for the given requirement description.\n"
                    # f"Step 4: Check whether your answer is right or not.\n"
                    # f"Step 5: Take the similar requirements description shown in the example to answer this question.\n"
                    f"###Requirement Description###\n{requirement}\n"
                    # f"###Current Chosen Path###\n{father_path}\n"
                    f"###Options###\n{child_node_str}\n"
                    f"Please pick up to {top_k} sub-options, your answer format MUST be like this:\n"
                    f"Answer: A, C, D\n"
                    f"Scores: A(0.9), C(0.7), D(0.5)\n"
                    # f"If none of the above options are correct, your answer should be: None.\n"
                    f"###Answer###"
                )

    # 合成 messages
    messages = [
        {"role": "system", "content": "You are an expert in requirements classification in the aerospace domain."},
        {"role": "user", "content": user_prompt},
    ]

    # 调用模型
    model_answer = call_model_api(messages)


    # 解析多选
    chosen_list = parse_multiple_answers(model_answer, all_leaf_paths)

    # 选置信度最高的 top_k_per_level
    chosen_list = sorted(chosen_list, key=lambda x: x[1], reverse=True)[:top_k]
     
    # 截取前三
    return chosen_list, user_prompt, model_answer



def batch_generate_answer_api(
    sentences: List[str],
    sampled_data: List[Dict],
    sampled_embeddings: List,
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
        chosen_list, user_prompt, model_answer = one_shot_classify(
            requirement=s,
            merged_tree_file=merged_tree,
            sampled_data=sampled_data,
            sampled_embeddings=sampled_embeddings,
            top_k=3
        )

       # 整理输出
        # chosen_list 里每个元素是 (路径, 置信度)
        # 例如 [("root-功能需求-性能需求", 0.9), ("root-接口需求-xxx", 0.7), ...]
        top_3_paths_str = []
        for path_str, conf in chosen_list:
            top_3_paths_str.append(f"{path_str} (conf={conf:.3f})")

        final_results.append(top_3_paths_str)

        # 历史对话只存一轮
        this_history = [
            ("User Prompt", user_prompt),
            ("Assistant Answer", model_answer)
        ]
        chat_history.append(this_history)

        # 更新 memory
        new_dict['Llm inference'] = top_3_paths_str
        new_dict['Correct answer'] = batch_answer[s_idx]
        memory_total.append(new_dict)

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
    answer_json_path = 'Results/ToT0315/answer_shot_'+str(LLM_MODEL)+'.json'
    memory_json_path = 'Results/ToT0315/memory_shot_'+str(LLM_MODEL)+'.json'
    embedding_path = 'Results/ToT0315/embedding_shot_'+str(LLM_MODEL)+'.npy'

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
    memory_finally, embedding_finally = load_memory_and_embedding(memory_json_path, embedding_path)
    # memory_finally = []
    # embedding_finally = []
    print(f'历史memory加载 {len(memory_finally)} 条')
    print(f'历史embedding加载 {len(embedding_finally)} 条')
    with open(sampled_data_path, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)
    sampled_embeddings = np.load(sampled_embeddings_path)
    print(f'samples加载 {len(sampled_data)} 条')
    print(f'embedding加载 {len(sampled_embeddings)} 条')
    

    # 读取未处理的requirements
    # unprocessed_data = load_unprocessed_requirements(excel_path, processed_count)
    unprocessed_data = json_to_dataframe(json_path, processed_count)
    print(f'待处理 {len(unprocessed_data)} 条')
    
    for batch_start in tqdm(range(0, len(unprocessed_data), eval_batch_size), desc="Generating outputs"):
        batch = unprocessed_data['Requirements Description'][batch_start: batch_start + eval_batch_size].tolist()
        batch_answer = unprocessed_data['Labels'][batch_start: batch_start + eval_batch_size].tolist()

        # 这里 batch_answer 是个列表，其中每个元素也是列表
        # 如果原先只存了一个路径，就类似 [["功能需求-数据需求"], ["接口需求-..."], ...]
        # responses, memory_total, chat_history = batch_generate_answer_api(
        #     batch,
        #     sampled_data,
        #     sampled_embeddings,
        #     merged_tree,
        #     batch_answer,
        #     max_depth=5
        # )
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