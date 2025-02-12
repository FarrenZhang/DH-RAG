import json
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("bigbio/pubhealth")

# Define the path to save the combined JSONL file
output_file = "pubhealth_test_validation.jsonl"

# Write the combined test and validation splits to a JSONL file
with open(output_file, 'w') as file:
    for split in ["test", "validation"]:
        for record in dataset[split]:
            json_record = json.dumps(record)
            file.write(json_record + "\n")

print(f"Test and validation dataset saved to {output_file}")

