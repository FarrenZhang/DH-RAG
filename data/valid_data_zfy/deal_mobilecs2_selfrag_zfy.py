import json
import argparse

def preprocess_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    processed_data = []
    for key, value in data.items():
        conversation = value['log']
        for turn in conversation:
            if turn['user'] != 'Manual':
                item = {
                    "id": key,
                    "instruction": turn['user'],
                    "output": turn['system'],
                    "input": ""  # You can modify this if needed
                }
                processed_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Preprocess conversation data for the main script.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the preprocessed JSON file')
    
    args = parser.parse_args()
    
    preprocess_data(args.input_file, args.output_file)
    print(f"Preprocessing complete. Output saved to {args.output_file}")

if __name__ == "__main__":
    main()