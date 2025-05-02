import openai
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = "sk-hNBVKkscViQNxdSW309c2495F25343F2822bB40cA9B6D18b"
openai.api_base = "https://burn.hair/v1"

class QueryMatcher:
    def __init__(self, input_files, output_folder):
        self.input_files = input_files
        self.output_folder = output_folder

    def load_historical_data(self):
        file_path = self.input_files
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def completions_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def call_model_chatgpt(self, prompt, model, max_tokens=50):
        results = self.completions_with_backoff(
            model=model,
            messages=[
                {"role": "user",
                 "content": prompt},
            ],
            request_timeout=60,
            max_tokens=max_tokens,
        )
        result = results["choices"][0]["message"]["content"]
        return result

    def query_gpt3(self, prompt):
        print(f"Querying GPT-3 model with prompt: {prompt}")
        response = self.call_model_chatgpt(
            prompt, model="gpt-3.5-turbo", max_tokens=100)
        print("GPT-3 model query completed")
        if hasattr(response, 'choices'):
            return response.choices[0].text.strip()
        else:
            return response

    def generate_function_tags(self, docs):
        tags = {}
        for doc_id, content in docs.items():
            print(f"Processing document ID: {doc_id}")
            prompt = f"Please describe how this document can help answer related questions: {content}"
            tags[doc_id] = self.query_gpt3(prompt)
            print(f"Tag generated for document ID: {doc_id}")
        print("Tags generated for all documents")
        return tags

    def extract_functional_requirements(self, query):
        prompt = f"What help is needed for this new query '{query}'? Please provide a detailed analysis."
        return self.query_gpt3(prompt)

    def similarity(self, tag, needed_help):
        tag_vector = np.array(tag)
        needed_help_vector = np.array(needed_help)
        dot_product = np.dot(tag_vector, needed_help_vector)
        norm_tag = np.linalg.norm(tag_vector)
        norm_needed_help = np.linalg.norm(needed_help_vector)
        similarity_score = dot_product / (norm_tag * norm_needed_help)
        return similarity_score

    def match_documents(self, function_tags, needed_help):
        best_match = None
        best_score = -1
        for doc_id, tag in function_tags.items():
            score = self.similarity(tag, needed_help)
            if score > best_score:
                best_match = doc_id
                best_score = score
        return best_match

    def calculate_similarity(self, new_query, historical_query):
        vectorizer = TfidfVectorizer().fit_transform([new_query, historical_query])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        return similarity[0][0]

    def main(self, new_query):
        historical_docs = self.load_historical_data()
        needed_help = self.extract_functional_requirements(new_query)
        print(f"Needed help: {needed_help}")
        topN = 3
        similarities = []
        top_queries = []
        for dialogue in historical_docs['dialogue']:
            if dialogue['speaker'] == 'User Zhang':
                historical_query = dialogue['query']
                similarity = self.calculate_similarity(new_query, historical_query)
                similarities.append(similarity)
                top_queries.append(historical_query)
        sorted_queries = [query for _, query in sorted(zip(similarities, top_queries), reverse=True)]
        print(f"Current query: {new_query}")
        print(f"Top {topN} queries with highest similarity:")
        for i in range(topN):
            print(f"Query {i+1}: {sorted_queries[i]} (Similarity: {similarities[i]})")
        average_similarity = sum(similarities) / len(similarities)
        print(f"Average similarity: {average_similarity}")

if __name__ == "__main__":
    matcher = QueryMatcher("./input_data/test_input.json", "./output_data/test_output.txt")
    matcher.main(sys.argv[1])
