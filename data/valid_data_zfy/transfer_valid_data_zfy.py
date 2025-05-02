import json

def transform_format(input_file, output_file):
    # Load the original JSON data from a file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Initialize the new structure
    transformed_data = {}

    # Assuming there's only one key in the outermost dictionary
    key = next(iter(data))
    chat_log = data[key]['log']

    # Prepare the new format
    transformed_data[key] = {"log": []}
    for entry in chat_log:
        user_message = entry.get("user", "")
        system_message = entry.get("system", "")
        api_query = entry.get("api_query", "")
        api_result = entry.get("api_result", "")

        # Create a log entry in the new format
        new_log_entry = {
            "user": user_message,
            "system": system_message,
            "api_query": api_query,
            "api_result": api_result
        }
        transformed_data[key]['log'].append(new_log_entry)

    # Write the transformed data to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(transformed_data, file, indent=4)

# Example usage
transform_format('input.json', 'output.json')
