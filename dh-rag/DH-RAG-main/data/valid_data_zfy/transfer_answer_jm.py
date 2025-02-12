import json

# Function to read the JSON file line by line
def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

# Load the test output JSON file line by line
test_output_data = read_json_file('test_output0805.json')

# Load the transformed conversations JSONL file line by line
transformed_conversations = read_json_file('transformed_conversations_jm0806.jsonl')

# Create a dictionary for quick lookup of answers by claim
answers_dict = {conv["claim"]: conv["answers"] for conv in transformed_conversations}

# Fill the "answers" field in the test output data
for item in test_output_data:
    claim = item["claim"]
    if claim in answers_dict:
        item["answers"] = answers_dict[claim]

# Save the modified test output data back to a JSON file
with open('test_output0805_filled.json', 'w') as output_file:
    for item in test_output_data:
        output_file.write(json.dumps(item, ensure_ascii=False, indent=4) + '\n')

print("The answers field has been filled successfully and saved to test_output0805_filled.json")
