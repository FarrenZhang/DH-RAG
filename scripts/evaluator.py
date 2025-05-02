import json
import logging
import os
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.ensure_nltk_resources()

    def ensure_nltk_resources(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logging.info("Downloading NLTK 'punkt' resource...")
            nltk.download('punkt', quiet=True)
            logging.info("Download complete.")

    def perform_bleu_evaluation(self):
        gpt3_answers = []
        standard_answers = []
        queries = []
        prompts = []
        
        self.load_answers(gpt3_answers, standard_answers, queries, prompts)
        
        if not gpt3_answers:
            logging.error("No answers loaded for evaluation.")
            return 0
        
        gpt3_answers_tokenized = [word_tokenize(answer.lower()) for answer in gpt3_answers]
        standard_answers_tokenized = [word_tokenize(answer.lower()) for answer in standard_answers]

        bleu_scores = self.calculate_bleu_scores(gpt3_answers_tokenized, standard_answers_tokenized)

        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        logging.info(f"Number of answer pairs evaluated: {len(bleu_scores)}")
        logging.info(f"Average BLEU score: {average_bleu:.4f}")

        self.save_detailed_results(queries, prompts, gpt3_answers, standard_answers, bleu_scores)

        return average_bleu

    def load_answers(self, gpt3_answers, standard_answers, queries, prompts):
        try:
            with open(self.config.GPT3_ANSWERS_FILE, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    gpt3_answers.append(data['gpt3_answer'])
                    standard_answers.append(data['context'])  # Using context as standard answer
                    queries.append(data['query'])
                    prompts.append(data['prompt'])
            logging.info(f"Loaded {len(gpt3_answers)} answer pairs for evaluation.")
        except FileNotFoundError:
            logging.error(f"GPT3 answers file not found: {self.config.GPT3_ANSWERS_FILE}")
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON in GPT3 answers file: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error reading GPT3 answers file: {str(e)}")

    def calculate_bleu_scores(self, gpt3_answers_tokenized, standard_answers_tokenized):
        bleu_scores = []
        for gpt3_answer, standard_answer in zip(gpt3_answers_tokenized, standard_answers_tokenized):
            try:
                bleu_score = sentence_bleu([standard_answer], gpt3_answer)
                bleu_scores.append(bleu_score)
            except Exception as e:
                logging.error(f"Error calculating BLEU score: {str(e)}")
        return bleu_scores

    def save_detailed_results(self, queries, prompts, gpt3_answers, standard_answers, bleu_scores):
        results_file = os.path.join(self.config.OUTPUT_DIR, "bleu_evaluation_results.jsonl")
        try:
            with open(results_file, 'w') as f:
                for i, (query, prompt, gpt3_answer, standard_answer, bleu_score) in enumerate(zip(queries, prompts, gpt3_answers, standard_answers, bleu_scores)):
                    result = {
                        "id": i,
                        "query": query,
                        "prompt": prompt,
                        "gpt3_answer": gpt3_answer,
                        "standard_answer": standard_answer,
                        "bleu_score": bleu_score
                    }
                    json.dump(result, f)
                    f.write('\n')
            logging.info(f"Detailed BLEU evaluation results written to {results_file}")
        except IOError as e:
            logging.error(f"IOError writing BLEU evaluation results: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error writing BLEU evaluation results: {str(e)}")

    def calculate_rouge_scores(self, gpt3_answers, standard_answers):
        # This is a placeholder for ROUGE score calculation
        # You would need to implement or use a library for ROUGE scoring
        logging.info("ROUGE score calculation not implemented.")
        return []

    def calculate_meteor_scores(self, gpt3_answers, standard_answers):
        # This is a placeholder for METEOR score calculation
        # You would need to implement or use a library for METEOR scoring
        logging.info("METEOR score calculation not implemented.")
        return []

    def perform_comprehensive_evaluation(self):
        gpt3_answers = []
        standard_answers = []
        queries = []
        prompts = []
        
        self.load_answers(gpt3_answers, standard_answers, queries, prompts)
        
        if not gpt3_answers:
            logging.error("No answers loaded for evaluation.")
            return {}
        
        bleu_scores = self.calculate_bleu_scores(
            [word_tokenize(answer.lower()) for answer in gpt3_answers],
            [word_tokenize(answer.lower()) for answer in standard_answers]
        )
        
        rouge_scores = self.calculate_rouge_scores(gpt3_answers, standard_answers)
        meteor_scores = self.calculate_meteor_scores(gpt3_answers, standard_answers)
        
        results = {
            "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
            "rouge": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
            "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
        }
        
        logging.info(f"Comprehensive evaluation results: {results}")
        
        return results