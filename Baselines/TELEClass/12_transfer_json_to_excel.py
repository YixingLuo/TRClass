import json
import pandas as pd
from typing import List, Dict

# 读取JSON文件
def read_json(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return []

# 保存到Excel文件
def save_to_excel(data: List[Dict], file_path: str):
    try:
        # 定义列名
        columns = ['req', '初始化', '可复用需求', '可维护需求', '可靠性需求', '姿态控制', '姿态确定', '安全性需求',
                   '控制输出', '故障诊断', '数据有效性判断', '数据采集', '时间约束', '模式管理', '空间约束',
                   '精度约束', '轨道计算', '遥控', '遥测']
        
        # 创建数据列表
        rows = []
        for item in data:
            req = item['req']
            final_classes = item['最终的class']
            row = [req] + [0] * 18
            for i, class_name in enumerate(columns[1:], 1):  
                if class_name in final_classes:
                    row[i] = 1
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=columns)
        
        df.to_excel(file_path, index=False)
        print(f"saved to {file_path}")
    except Exception as e:
        print(f"save {file_path} error: {e}")

def process_class_to_binary_matrix(input_file: str, output_file: str):
    input_data = read_json(input_file)
    save_to_excel(input_data, output_file)

if __name__ == "__main__":
    input_file_path = r"path/to/refined_core_classes.json"
    output_file_path = r"path/to/refined_core_classes.xlsx"
    process_class_to_binary_matrix(input_file_path, output_file_path)