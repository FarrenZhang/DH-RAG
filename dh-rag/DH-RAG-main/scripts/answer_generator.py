import json
import logging
import time
import openai

class AnswerGenerator:
    def __init__(self, config):
        self.config = config
        openai.api_base = config.OPENAI_API_BASE
        openai.api_key = config.OPENAI_API_KEY

    def generate_answers(self, integrated_results):
        logging.info("Starting to generate GPT-3.5-turbo answers...")
        
        gpt3_answers = []

        for item in integrated_results:
            query = item['query']
            context = item['integrated_results']

            prompt = f"""
            Query: {query}

            Context: {context}

            Based on the above information, please provide a comprehensive answer to the query.
            """
            logging.info(f"Prompt for query '{query[:50]}...': {prompt}")

            try:
                time.sleep(1)  # To avoid OpenAI API rate limits
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    n=1,
                    temperature=0.7,
                )
                gpt3_answer = response.choices[0].message['content'].strip()
                logging.info(f"Generated GPT-3.5-turbo answer for query: {gpt3_answer[:50]}...")
                
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI API error for query '{query[:50]}...': {str(e)}")
                gpt3_answer = f"Unable to generate answer due to API issue: {str(e)}"
            except Exception as e:
                logging.error(f"Unexpected error in GPT-3.5-turbo call for query '{query[:50]}...': {str(e)}")
                gpt3_answer = f"Unable to generate answer due to unexpected issue: {str(e)}"

            gpt3_answers.append({
                "query": query,
                "prompt": prompt,
                "gpt3_answer": gpt3_answer,
                "context": context
            })

            self.save_answer(gpt3_answers[-1], query)

        logging.info(f"All GPT-3.5-turbo answers processed. Total: {len(gpt3_answers)}")
        return gpt3_answers

    def save_answer(self, answer, query):
        try:
            with open(self.config.GPT3_ANSWERS_FILE, 'a') as f:
                json.dump(answer, f)
                f.write('\n')
            logging.info(f"Answer for query '{query[:50]}...' written to {self.config.GPT3_ANSWERS_FILE}")
        except Exception as e:
            logging.error(f"Error writing answer for query '{query[:50]}...' to file: {str(e)}")