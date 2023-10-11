# log_config.py

import logging
import datetime


def setup_logging():
    """Sets up logging configuration."""
    # Define the structured format for the logs
    log_format = (
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - " "%(message)s"
    )

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

    info_file_handler = logging.FileHandler("simulation.log")
    info_file_handler.setLevel(logging.INFO)  # Log everything to the file
    info_file_handler.setFormatter(logging.Formatter(log_format))

    # Add the handlers to the logger
    logger = logging.getLogger()
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(info_file_handler)

    return logger


def log_config(config: dict):
    """Logs the provided configuration."""
    logger = logging.getLogger()
    formatted_config = "\n".join([f"{key}: {value}" for key, value in config.items()])
    logger.info(f"Simulation Configuration:\n{formatted_config}")


def log_simulation_start():
    logger = logging.getLogger()
    logger.info("------ Start of New Simulation: %s ------", datetime.datetime.now())


def print_to_log(*args, **kwargs):
    """Custom print function to log instead of printing to stdout."""
    logger = logging.getLogger()
    # logger.info(" ".join(map(str, args)))
    logger.info(*args, **kwargs)
