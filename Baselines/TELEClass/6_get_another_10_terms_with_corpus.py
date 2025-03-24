import json
from typing import List, Dict
from openai import OpenAI
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import tiktoken

def read_json(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return []

def save_json(data: List[Dict], file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"saved to {file_path}")
    except Exception as e:
        print(f"save {file_path} error: {e}")

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    return len(encoding.encode(text))

# Use LexRank to compress requirements
def compress_requirements(reqs_str: str, target_tokens: int = 15000, max_tokens: int = 16385) -> str:
    parser = PlaintextParser.from_string(reqs_str, Tokenizer("chinese"))
    summarizer = LexRankSummarizer()
    
    total_sentences = len(parser.document.sentences)
    
    min_sentences = 1
    max_sentences = total_sentences
    compressed_reqs = reqs_str
    current_tokens = count_tokens(reqs_str)
    
    if current_tokens <= target_tokens:
        return reqs_str
    
    while min_sentences <= max_sentences:
        mid_sentences = (min_sentences + max_sentences) // 2
        summary = summarizer(parser.document, mid_sentences)
        compressed_reqs = ", ".join(str(sentence) for sentence in summary)
        current_tokens = count_tokens(compressed_reqs)
        
        if current_tokens > max_tokens:
            max_sentences = mid_sentences - 1
        elif current_tokens > target_tokens:
            max_sentences = mid_sentences - 1
        elif current_tokens < target_tokens - 1000:  
            min_sentences = mid_sentences + 1
        else:
            break
    
    return compressed_reqs

def ask_llm(prompt: str, index: int):
    client = OpenAI(
        api_key="",  
        base_url="https://api.openai-proxy.org/v1",
    )
    token_count = count_tokens(prompt)
    if token_count > 16385:
        return ""
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        return ""

def process_class_terms(input_file: str, output_file: str):
    input_data = read_json(input_file)
    
    new_data = []
    
    for index, item in enumerate(input_data, start=1):
        class_name = item['class']
        reqs = item['req']
        
        reqs_str = ", ".join(reqs)
        
        compressed_reqs = compress_requirements(reqs_str, target_tokens=12000)
        
        prompt = f"""
        You are an expert in aerospace control software requirements. Given the following class and its associated requirements, please choose the 10 most critical key phrases (not sentences) that best represent this class from the requirements. Key phrases should be concise noun phrases or verb-noun pairs (e.g., '姿态控制', '数据采集'). Return only the key phrases, one per line.

        Class: {class_name}
        Requirements: {compressed_reqs}
        """
        
        response = ask_llm(prompt, index)
        print(f"response：{response}")
        if not response:
            print(f"class{index}: {class_name} fails，step over")
            continue
        
        key_phrases = [line.strip() for line in response.split('\n') if line.strip()]
        key_phrases = key_phrases[:10]  
        
        new_item = {
            "class": class_name,
            "terms": key_phrases
        }
        new_data.append(new_item)
    
    save_json(new_data, output_file)

if __name__ == "__main__":
    input_file_path = r"path/to/reqs_under_classes.json"
    output_file_path = r"path/to/another_10_terms_with_corpus.json"
    process_class_terms(input_file_path, output_file_path)