# log_config.py

import logging


def setup_logging():
    """Sets up logging configuration."""
    # Define the structured format for the logs
    log_format = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    # Set up the root logger
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Create a file handler that logs everything to a file
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create a stream handler to log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)  # Only log errors to stdout for pytest
    stream_handler.setFormatter(logging.Formatter(log_format))

    # Add the handlers to the logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def print_to_log(*args, **kwargs):
    """Custom print function to log instead of printing to stdout."""
    logger = logging.getLogger()
    logger.info(" ".join(map(str, args)))
