import json

# Define the input and output file paths
input_file_path = 'health_claims_processed.jsonl'
output_file_path = 'health_claims_top25.jsonl'

# Read the data from the input file
data = []
with open(input_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

# Extract the top 10 entries
top_10_data = data[:25]

# Write the top 10 entries to a new JSONL file
with open(output_file_path, 'w') as file:
    for entry in top_10_data:
        file.write(json.dumps(entry) + '\n')

print(f"Top 25 entries have been written to {output_file_path}")
