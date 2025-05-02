import json

# Function to read data from a JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    return data

# Function to write data to a JSONL file
def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# Function to load the valid conversations JSON file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to find the system response for a given user query
def find_system_response(valid_conversations, user_query):
    for key, value in valid_conversations.items():
        for log in value['log']:
            if log['user'] == user_query:
                return log['system']
    return ""

# Function to restructure the data
def restructure_data(data, valid_conversations):
    restructured_data = []
    for item in data:
        query = item.get('query', '')
        ctxs = []
        for result in item.get('RAG_results', []):
            ctx = {
                "id": result.get('id', ''),
                "title": result.get('title', ''),
                "text": result.get('text', ''),
                "score": result.get('score', 'UNKNOWN')  # Use "UNKNOWN" as default if score is not available
            }
            ctxs.append(ctx)
        
        system_response = find_system_response(valid_conversations, query)
        
        new_entry = {
            "claim": query,
            "label": "",  # Placeholder, adjust as needed
            "question": query,
            "answers": [system_response if system_response else ""],  # Use the found system response or a placeholder
            "output": [""],  # Placeholder, adjust as needed
            "ctxs": ctxs
        }
        restructured_data.append(new_entry)
    return restructured_data

# File paths
input_file_path = 'conversations_with_retrieval.jsonl'
output_file_path = 'transformed_conversations_jm0806.jsonl'
answer_file_path = 'valid_conversations.json'

# Read the original data
original_data = read_jsonl(input_file_path)
valid_conversations = read_json(answer_file_path)

# Restructure the data
restructured_data = restructure_data(original_data, valid_conversations)

# Write the restructured data to a new JSONL file
write_jsonl(output_file_path, restructured_data)

print(f"Restructured data has been saved to {output_file_path}")
