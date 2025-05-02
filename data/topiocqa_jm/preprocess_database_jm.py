import json
import hashlib

def preprocess_data(input_file, output_file, mapping_file):
    all_contexts = {}
    query_context_mapping = {}

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            conversation_no = data['Conversation_no']
            turn_no = data['Turn_no']
            query_id = f"{conversation_no}_{turn_no}"
            query_context_mapping[query_id] = []

            if 'Gold_passage' in data:
                gold_passage = data['Gold_passage']
                text = gold_passage.get('text', '')
                title = gold_passage.get('title', '')

                # Create a unique ID for each context
                ctx_id = hashlib.md5(text.encode()).hexdigest()

                if ctx_id not in all_contexts:
                    all_contexts[ctx_id] = {
                        'id': ctx_id,
                        'text': text,
                        'title': title
                    }

                query_context_mapping[query_id].append(ctx_id)

    # Write all unique contexts to the output file
    with open(output_file, 'w') as f:
        for ctx in all_contexts.values():
            json.dump(ctx, f)
            f.write('\n')

    # Write the query-context mapping
    with open(mapping_file, 'w') as f:
        json.dump(query_context_mapping, f)

# Run the preprocessing
preprocess_data('topiocqa_validation_first_16_convo.jsonl', 'retrieval_database.jsonl', 'query_context_mapping.json')