import json
from typing import List, Dict
from openai import OpenAI

def read_json(file_path: str) -> Dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"read {file_path} error: {e}")
        return {}

def ask_llm(prompt, index):
    client = OpenAI(
        api_key="",
        base_url="https://api.openai-proxy.org/v1",
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are an expert in classification for aerospace control software requirements."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"req{index} error:{e}")
        return ""

def extract_classes(llm_response: str, available_classes: List[str]) -> List[str]:
    lines = llm_response.split('\n')
    selected_classes = []
    for line in lines:
        cleaned = line.strip('- ').strip()
        if cleaned in available_classes and cleaned not in selected_classes:
            selected_classes.append(cleaned)
            if len(selected_classes) == 3:
                break
    return selected_classes[:3] 

def process_requirements(req_file: str, output_file: str):
    req_data = read_json(req_file)
    
    new_data = []
    for index, item in enumerate(req_data, start=1):
        print(f"req{index}")
        req = item['req']
        classes = item['classes']
        prompt = f"""You will be provided with an aerospace control software requirement. Please select the 3 most relevant requirement types from the following categories: {', '.join(classes)}. Sort them by relevance from highest to lowest. Just give the category names as shown in the provided list, one per line.
        Requirement: {req}
        """
        llm_response = ask_llm(prompt, index)
        if not llm_response:
            print(f"req{index}: '{req}' fails，step over")
            continue
        
        top_classes = extract_classes(llm_response, classes)
        
        new_item = {
            'req': req,
            'classes': classes,  
            'LLM_core_classes': top_classes  
        }
        new_data.append(new_item)
    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"saved to {output_file}")

if __name__ == "__main__":
    req_file_path = r"path/to/core_classes.json"  
    output_file_path = r"path/to/3_classes_from_core_classes.json"   
    process_requirements(req_file_path, output_file_path)