import json
import time
import openai
from openai.error import APIError, RateLimitError, InvalidRequestError, AuthenticationError

# Set your API key and API base
openai.api_base = "https://api.7xnn.cn/v1"
openai.api_key = "sk-D0fmelpB0obrIxiUBd47D6D629B0419fB4Ef42F85b5eF0F5"

# Load the original JSONL data
with open('health_claims_top10.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Function to call OpenAI API with a similar approach as run_mobilecs2_baseline_zfy.py
def call_openai_api(prompt, model="gpt-3.5-turbo", max_tokens=50, temperature=0.7, retries=3):
    print(f"Calling call_openai_api with model: {model}")
    print(f"Prompt: {prompt}")
    for i in range(retries):
        try:
            time.sleep(0.5)  # To avoid OpenAI API rate limits
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                n=1,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except (APIError, RateLimitError) as e:
            wait_time = 2 ** i  # Exponential backoff
            print(f"API error occurred: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except (InvalidRequestError, AuthenticationError) as e:
            print(f"Request error: {e}. This error is not retryable.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {2 ** i} seconds...")
            time.sleep(2 ** i)
    raise Exception("Failed to call OpenAI API after multiple retries.")

# Define a function to generate follow-up claims and contexts using GPT-3.5-turbo
def generate_follow_ups_and_contexts(claim, num_follow_ups=9, num_contexts=20):
    follow_up_claims = []
    for _ in range(num_follow_ups):
        follow_up_claim = call_openai_api(f"Generate a follow-up claim for the given claim: {claim}")
        follow_up_claims.append(follow_up_claim)

    follow_up_data = []
    for follow_up_claim in follow_up_claims:
        contexts = []
        for _ in range(num_contexts):
            context_text = call_openai_api(f"Generate a relevant context for the given claim: {follow_up_claim}")
            context = {"id": "", "title": "", "text": context_text}
            contexts.append(context)
        follow_up_data.append({"claim": follow_up_claim, "ctxs": contexts})
    
    return follow_up_data

# Process each claim in the original data
new_data = []
for entry in data:
    original_claim = entry['claim']
    original_contexts = entry['ctxs']
    
    # Add the original claim and contexts to the new data
    new_data.append({"claim": original_claim, "ctxs": original_contexts})
    
    # Generate follow-up claims and contexts
    try:
        follow_up_data = generate_follow_ups_and_contexts(original_claim)
        # Add the follow-up claims and contexts to the new data
        new_data.extend(follow_up_data)
    except Exception as e:
        print(f"Failed to generate follow-ups for claim: {original_claim}. Error: {e}")

# Save the new data to a JSONL file
with open('health_claims_top10_follow_ups.jsonl', 'w') as f:
    for entry in new_data:
        f.write(json.dumps(entry) + '\n')

print("New JSONL file with follow-up claims and contexts created successfully.")

