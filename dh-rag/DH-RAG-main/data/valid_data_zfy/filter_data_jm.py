import json

# Define the function to read JSON file line by line
def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    return data

# Try to load the JSON file as a single JSON array
try:
    with open('test_output0805_filled.json', 'r') as file:
        data = json.load(file)
except json.JSONDecodeError:
    # If it fails, load the JSON file line by line
    data = read_json_file('test_output0805_filled.json')

# Filter out the items with the specific output message
filtered_data = [item for item in data if "由于API问题无法生成答案..." not in item.get("output", "")]

# Save the filtered data back to a new JSON file
with open('test_output0805_filtered.json', 'w') as output_file:
    for item in filtered_data:
        output_file.write(json.dumps(item, ensure_ascii=False, indent=4) + '\n')

print(f"Filtered data saved to test_output0805_filtered.json. Removed {len(data) - len(filtered_data)} items.")
