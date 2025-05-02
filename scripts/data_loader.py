import json
import logging
from collections import namedtuple
import os

ConversationData = namedtuple('ConversationData', ['history', 'query'])

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        try:
            logging.info(f"Attempting to load data from: {self.config.INPUT_DATA_FILE}")
            
            if not os.path.exists(self.config.INPUT_DATA_FILE):
                logging.error(f"Input file does not exist: {self.config.INPUT_DATA_FILE}")
                return None

            with open(self.config.INPUT_DATA_FILE, 'r', encoding='utf-8') as file:
                data = json.load(file)

            paired_data = []
            qa_count = 0
            total_conversations = len(data)
            history_lengths = []

            for conversation_id, conversation_data in data.items():
                qa_found = False
                for index, message in enumerate(conversation_data['log']):
                    if message['api_query'] == "[QA]":
                        query = message
                        history = conversation_data['log'][:index] + conversation_data['log'][index+1:]
                        history_length = len(history)
                        history_lengths.append(history_length)
                        paired_data.append(ConversationData(history, query))
                        qa_count += 1
                        qa_found = True
                        logging.info(f"Conversation {conversation_id}: History length = {history_length}")
                        break
                
                if not qa_found:
                    logging.info(f"Conversation {conversation_id} does not contain a [QA] item.")

            logging.info(f"Total conversations processed: {total_conversations}")
            logging.info(f"Total conversations with [QA] items: {qa_count}")
            logging.info(f"Percentage of conversations with [QA] items: {(qa_count/total_conversations)*100:.2f}%")
            
            if history_lengths:
                avg_history_length = sum(history_lengths) / len(history_lengths)
                max_history_length = max(history_lengths)
                min_history_length = min(history_lengths)
                logging.info(f"Average history length: {avg_history_length:.2f}")
                logging.info(f"Maximum history length: {max_history_length}")
                logging.info(f"Minimum history length: {min_history_length}")

            return paired_data
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {str(e)}")
            with open(self.config.INPUT_DATA_FILE, 'r', encoding='utf-8') as file:
                file_content = file.read()
                logging.error(f"File content (first 100 characters): {file_content[:100]}")
        except Exception as e:
            logging.error(f"Error loading valid data: {str(e)}")
        
        return None