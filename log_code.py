import logging
import os

def setup_logging(script_name):
    logger = logging.getLogger(script_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Create a file handler for the script
        log_dir = f'D:\Customer_Retention_Prediction_System\logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 2. Construct the full file path safely
        log_file_path = os.path.join(log_dir, f'{script_name}.log')

        # 3. Create the file handler
        # Using raw strings (r'') avoids "invalid escape sequence" warnings
        handler = logging.FileHandler(log_file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger