import json

# 读取原始数据文件
with open('global_kb_english.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 准备转换后的数据列表
transformed_data = []

# 遍历原始数据，转换格式
for i, text in enumerate(data, start=1):
    transformed_data.append({
        "id": str(i),
        "title": f"Business Name {i}",  # 由于原始数据中没有标题，这里使用一个简单的占位符
        "text": text
    })

# 保存转换后的数据到新文件
with open('transformed_data.jsonl', 'w', encoding='utf-8') as outfile:
    for item in transformed_data:
        json.dump(item, outfile)
        outfile.write('\n')  # 每个JSON对象后添加换行符，以符合JSONL格式