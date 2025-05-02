import json
import os
from openai import OpenAI
import backoff
from openai import OpenAIError


# python generate_follow_ups_jm.py

# Set up your OpenAI API key
client = OpenAI(api_key="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")

input_file = 'extracted_train_samples_0808.jsonl'
output_file = 'popqa_train_0808.jsonl'

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def call_model_chatgpt(prompt, model, max_tokens=50):
    results = completions_with_backoff(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return results.choices[0].message.content

with open(input_file, 'r') as file:
    data = [json.loads(line) for line in file]

def generate_follow_up_set(original_question, original_answers, original_ctxs):
    # Use OpenAI API to generate follow-up question, contexts, and answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a relevant follow-up set based on the given question. Include a follow-up question, 10 relevant context entries with id, title, and text, and an answer. Format your response as JSON with keys 'follow_up_question', 'follow_up_ctxs', and 'answer'."},
            {"role": "user", "content": f"Question: {original_question}\nAnswers: {original_answers}\nContexts: {json.dumps(original_ctxs)}"}
        ]
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        follow_up_question = result['follow_up_question']
        follow_up_ctxs = result['follow_up_ctxs']
        answer = result['answer']
    except json.JSONDecodeError:
        # If JSON parsing fails, use a fallback method
        content = response.choices[0].message.content
        follow_up_question = content.split('\n')[0].strip()
        follow_up_ctxs = [{"id": "fallback", "title": "Fallback Context", "text": content}]
        answer = "Unable to generate a specific answer due to parsing error."
    
    return follow_up_question, follow_up_ctxs, answer

def generate_follow_ups(data_set, num_follow_ups=9):
    all_entries = [data_set]  # Start with the original entry
    
    # Generate follow-ups
    for _ in range(num_follow_ups):
        follow_up_question, follow_up_ctxs, answer = generate_follow_up_set(data_set['question'], data_set['answers'], data_set['ctxs'])
        follow_up_entry = {
            "question": follow_up_question,
            "answers": [answer],  # List containing the generated answer
            "prop": data_set.get('prop', ''),  # Preserve the 'prop' field if it exists
            "s_wiki_title": data_set.get('s_wiki_title', ''),  # Preserve the 's_wiki_title' field if it exists
            "id": data_set.get('id', ''),  # Preserve the 'id' field if it exists
            "pop": data_set.get('pop', ''),  # Preserve the 'pop' field if it exists
            "ctxs": follow_up_ctxs
        }
        all_entries.append(follow_up_entry)
    return all_entries

# Process each set in the input data
all_sets = []
for data_set in data:
    all_sets.extend(generate_follow_ups(data_set))

# Save the original data and follow-ups to a JSONL file
with open(output_file, 'w') as file:
    for entry in all_sets:
        file.write(json.dumps(entry) + '\n')

print(f"Original questions and generated follow-up questions saved to {output_file}")