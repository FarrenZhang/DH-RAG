import json
import itertools
import random

def group_queries(input_data):
    grouped_data = {}
    conversation_id = 1

    # Group queries by subject (assuming subject changes every ~10 queries)
    for key, group in itertools.groupby(enumerate(input_data), lambda x: x[0] // 10):
        conversation = {
            "log": []
        }
        
        group_list = list(group)
        total_queries = len(group_list)
        
        # Determine which of the last 5 queries (or fewer if less than 5) will be QA
        qa_indices = random.sample(range(max(0, total_queries - 5), total_queries), min(2, total_queries))
        
        for idx, (_, item) in enumerate(group_list):
            is_qa = idx in qa_indices
            conversation["log"].append({
                "user": item["question"],
                "system": item["answers"][0] if isinstance(item["answers"], list) else item["answers"],
                "api_query": "[QA]" if is_qa else ""
               # "api_result": ""
            })
        
        grouped_data[f"conversation_{conversation_id}"] = conversation
        conversation_id += 1

    return grouped_data

# Read the input data from JSONL file
input_data = []
with open('triviaqa_test_0808.jsonl', 'r') as f:
    for line in f:
        input_data.append(json.loads(line.strip()))

# Group the queries
grouped_data = group_queries(input_data)

# Write the grouped data to a new JSON file
with open('transformed_triviaqa_0812.json', 'w') as f:
    json.dump(grouped_data, f, indent=2)

print("Grouped conversations have been written to 'transformed_triviaqa_0812.json'")