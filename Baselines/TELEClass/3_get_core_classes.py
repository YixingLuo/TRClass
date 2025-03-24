import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained(r'path/to/bert-base-chinese')
model = AutoModel.from_pretrained(r'path/to/bert-base-chinese')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = model.to(device)
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

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()[0]
def compute_similarity(text1, text2):
    emb1 = get_sbert_embedding(text1)
    emb2 = get_sbert_embedding(text2)
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity

def read_requirements(file_path):
    df = pd.read_excel(file_path)
    requirements = df.iloc[:, 0].tolist()
    return requirements

def read_class_terms(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        class_terms = json.load(f)
    return class_terms

def compute_class_similarities(requirement, class_terms):
    class_similarities = []
    
    for class_term in class_terms:
        class_name = class_term["class"]
        print(f"calculate the similarity of req and {class_name}")
        terms = class_term["terms"]
        
        if not terms:
            continue
        
        max_similarity = -1
        for term in terms:
            similarity = compute_similarity(requirement, term)
            max_similarity = max(max_similarity, similarity)
        
        class_similarities.append((class_name, max_similarity))
    
    class_similarities.sort(key=lambda x: x[1], reverse=True)
    return class_similarities

def process_requirements_to_json(req_file, class_terms_file, output_json_file):
    requirements = read_requirements(req_file)
    class_terms = read_class_terms(class_terms_file)
    
    result = []
    i=1
    for req in requirements:
        print(f"req{i}")
        i+=1
        class_similarities = compute_class_similarities(req, class_terms)
        top_10_classes = [class_name for class_name, _ in class_similarities[:10]]
        
        result.append({
            "req": req,
            "classes": top_10_classes
        })

        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"saved to {output_json_file}")
    
if __name__ == "__main__":
    req_file = r"path/to/true_labels.xlsx"  
    class_terms_file = r"path/to/llm_10_terms.json"  
    output_json_file = r"path/to/core_classes.json"  
    
    process_requirements_to_json(req_file, class_terms_file, output_json_file) 