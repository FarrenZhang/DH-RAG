import json
import hashlib

def preprocess_data(input_file, output_file, mapping_file):
    all_contexts = {}
    query_context_mapping = {}

    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['id']
            query_context_mapping[query_id] = []

            for ctx in data['ctxs']:
                # Create a unique ID for each context
                ctx_id = hashlib.md5(ctx['text'].encode()).hexdigest()

                if ctx_id not in all_contexts:
                    all_contexts[ctx_id] = ctx

                query_context_mapping[query_id].append(ctx_id)

    # Write all unique contexts to the output file
    with open(output_file, 'w') as f:
        for ctx_id, ctx in all_contexts.items():
            ctx['id'] = ctx_id
            json.dump(ctx, f)
            f.write('\n')

    # Write the query-context mapping
    with open(mapping_file, 'w') as f:
        json.dump(query_context_mapping, f)

# Run the preprocessing
preprocess_data('coqa_with_ctxs_0813.jsonl', 'retrieval_database.jsonl', 'query_context_mapping.json')