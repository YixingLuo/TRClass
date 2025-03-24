import json
import pandas as pd

# 1. 定义18个标签
LABEL = ["初始化","可复用需求","可维护需求","可靠性需求","姿态控制","姿态确定","安全性需求","控制输出","故障诊断","数据有效性判断","数据采集","时间约束","模式管理","空间约束","精度约束","轨道计算","遥控","遥测"]
# 2. 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 3. 将JSON转换为表格
def json_to_table(json_data, labels):
    # 初始化表格数据
    table_data = []
    
    # 遍历每个需求
    for item in json_data[:-1]:
        req = item["req"]
        req_labels = item["labels"]
        
        # 创建一行数据
        row = [req]  # 第一列是需求描述
        
        # 对每个标签，检查是否在req_labels中
        for label in labels:
            if label in req_labels:
                row.append(1)
            else:
                row.append(0)
        
        table_data.append(row)
    
    # 创建列名：第一列是"req"，后面是18个标签
    columns = ["req"] + labels
    
    # 创建DataFrame
    df = pd.DataFrame(table_data, columns=columns)
    return df

# 4. 保存为Excel文件
def save_to_excel(df, output_path):
    df.to_excel(output_path, index=False)
    print(f"表格已保存到 {output_path}")

# 5. 主函数
def process_json_to_table(json_path, output_excel_path):
    # 读取JSON
    json_data = read_json(json_path)
    
    # 转换为表格
    df = json_to_table(json_data, LABEL)
    
    # 验证：每行应该有3个1
    for i in range(len(df)):
        row_sum = df.iloc[i, 1:].sum()  # 跳过第一列（req）
        assert row_sum == 3, f"第 {i+1} 行有 {row_sum} 个1，期望是3个1"
    
    # 保存为Excel
    save_to_excel(df, output_excel_path)

# 6. 使用示例
if __name__ == "__main__":
    json_path = r"ReqMulti\原始数据\hier.json"  # 输入JSON文件路径
    output_excel_path = r"ReqMulti\原始数据\hier预测.xlsx"  # 输出Excel文件路径
    process_json_to_table(json_path, output_excel_path)