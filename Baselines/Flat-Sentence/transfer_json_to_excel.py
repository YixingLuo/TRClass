import json
import pandas as pd

LABEL = ["初始化","可复用需求","可维护需求","可靠性需求","姿态控制","姿态确定","安全性需求","控制输出","故障诊断","数据有效性判断","数据采集","时间约束","模式管理","空间约束","精度约束","轨道计算","遥控","遥测"]
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def json_to_table(json_data, labels):
    table_data = []
    
    for item in json_data[:-1]:
        req = item["req"]
        req_labels = item["labels"]
        
        row = [req] 
        
        for label in labels:
            if label in req_labels:
                row.append(1)
            else:
                row.append(0)
        
        table_data.append(row)
    
    columns = ["req"] + labels
    
    df = pd.DataFrame(table_data, columns=columns)
    return df

def save_to_excel(df, output_path):
    df.to_excel(output_path, index=False)
    print(f"saved to {output_path}")

def process_json_to_table(json_path, output_excel_path):
    json_data = read_json(json_path)
    
    df = json_to_table(json_data, LABEL)

    for i in range(len(df)):
        row_sum = df.iloc[i, 1:].sum() 
        assert row_sum == 3
    
    save_to_excel(df, output_excel_path)

if __name__ == "__main__":
    json_path = r"path/to/flatten.json"  
    output_excel_path = r"path/to/flatten.xlsx" 
    process_json_to_table(json_path, output_excel_path)