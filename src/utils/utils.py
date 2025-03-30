import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    This function checks whether the specified directory exists. If it does not,
    the directory is created. Logs are generated to indicate whether the directory
    was created or already exists.

    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")