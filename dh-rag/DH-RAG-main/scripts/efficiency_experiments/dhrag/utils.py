import json
import logging

def initialize_output_file(file_path):
    """Initialize (or overwrite) an output file"""
    try:
        with open(file_path, "w") as f:
            f.write("")  # Create an empty file
        logging.info(f"Initialized output file: {file_path}")
    except Exception as e:
        logging.error(f"Error initializing output file {file_path}: {str(e)}")

def save_json(data, file_path):
    """Save data as JSON to a file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {str(e)}")

def load_json(file_path):
    """Load JSON data from a file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return None

# Add more utility functions as needed