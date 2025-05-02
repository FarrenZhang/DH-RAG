import json
import os
from openai import OpenAI
import backoff
from openai import OpenAIError

# Command Line: python generate_follow_ups_jm.py

# Set up your OpenAI API key
client = OpenAI(api_key="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")

input_file = 'popqa_train_0809.jsonl'  # Your existing file with questions
output_file = 'popqa_train_0809_with_additional_followup.jsonl'

@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

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

def generate_one_follow_up(question_entry):
    follow_up_question, follow_up_ctxs, answer = generate_follow_up_set(
        question_entry['question'], 
        question_entry['answers'], 
        question_entry['ctxs']
    )
    
    follow_up_entry = {
        "question": follow_up_question,
        "answers": [answer],  # List containing the generated answer
        "prop": question_entry.get('prop', ''),  # Preserve the 'prop' field if it exists
        "s_wiki_title": question_entry.get('s_wiki_title', ''),  # Preserve the 's_wiki_title' field if it exists
        "id": question_entry.get('id', ''),  # Preserve the 'id' field if it exists
        "pop": question_entry.get('pop', ''),  # Preserve the 'pop' field if it exists
        "ctxs": follow_up_ctxs
    }
    return follow_up_entry

# Read the existing file
with open(input_file, 'r') as file:
    all_questions = [json.loads(line) for line in file]

# Generate one follow-up for each question
all_entries = []
for question in all_questions:
    all_entries.append(question)  # Keep the original question
    follow_up = generate_one_follow_up(question)
    all_entries.append(follow_up)  # Add the new follow-up

# Save the original questions and new follow-ups to a JSONL file
with open(output_file, 'w') as file:
    for entry in all_entries:
        file.write(json.dumps(entry) + '\n')

print(f"Original questions and one additional follow-up for each saved to {output_file}")