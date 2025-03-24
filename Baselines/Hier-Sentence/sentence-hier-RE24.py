import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json
from utils import *
import torch.nn.functional as F
import random
import time

LABEL = ["初始化","可复用需求","可维护需求","可靠性需求","姿态控制","姿态确定","安全性需求","控制输出","故障诊断","数据有效性判断","数据采集","时间约束","模式管理","空间约束","精度约束","轨道计算","遥控","遥测"]
SEED = 42   
random.seed(SEED)  
np.random.seed(SEED)  
torch.manual_seed(SEED)  
if torch.cuda.is_available():  
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
def load_tree_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def get_sbert_embedding(text):

    tokenizer = AutoTokenizer.from_pretrained(r'path/to/bert-base-chinese')
    model = AutoModel.from_pretrained(r'path/to/bert-base-chinese')

    
    model = model.to(device)
    model.eval() 
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()[0]

def calculate_similarity(input_text, categories):

    input_embedding_2 = get_sbert_embedding(input_text)
    category_embeddings = [get_sbert_embedding(category) for category in categories]
    similarities = [my_cosine_similarity(input_embedding_2, cat_emb) for cat_emb in category_embeddings]
    return similarities

def find_top_categories(tree, input_text, top_k=3):

    choices = [tree]

    while True:
        next_choices = []
        for node in choices:
            if "children" in node and node["children"]:
                next_choices.extend(node["children"])
            else:
                next_choices.append(node)

        if len(next_choices) == len(choices):
            break

        names = [node["name"] for node in next_choices]
        scores = calculate_similarity(input_text, names)

        sorted_indices = np.argsort(scores)[-top_k:][::-1]
        choices = [next_choices[i] for i in sorted_indices]
    return [node["name"] for node in choices]
start_time=time.time()
df = pd.read_excel(r'path/to/true_labels.xlsx')
tree=load_tree_from_json(r'path/to/hierarchy.json')
predictions=[]
if 'Requirements Description' in df.columns:
    descriptions = df['Requirements Description']
    for index, description in enumerate(descriptions,start=1):
        print(f'description{index}')
        labels = find_top_categories(tree, description, top_k=3)
        predictions.append({
            "req":description,
            "labels":labels
        })
        # break
end_time=time.time()
prediction_time=end_time-start_time
predictions.append({
    "start_time":start_time,
    "end_time":end_time,
    "time":prediction_time
})
with open(r'path/to/hier.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=2,ensure_ascii=False)