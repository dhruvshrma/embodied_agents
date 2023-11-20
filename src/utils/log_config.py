import logging
import datetime


def generate_filename_with_date(name: str, extension: str = ".log") -> str:
    today = datetime.datetime.now()
    formatted_date = today.strftime("%d_%m_%Y")
    return f"{name}_{formatted_date}{extension}"


def setup_logging(
    app_file_name: str = "app.log", simulation_file_name: str = "simulation.log"
):
    """Sets up logging configuration."""
    # Define the structured format for the logs
    log_format = (
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - " "%(message)s"
    )

    # Set up the root logger
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Create a file handler that logs everything to a file
    file_handler = logging.FileHandler(app_file_name)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create a stream handler to log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)  # Only log errors to stdout for pytest
    stream_handler.setFormatter(logging.Formatter(log_format))

    info_file_handler = logging.FileHandler(simulation_file_name)
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
    logger = logging.getLogger()
    formatted_config = "\n".join([f"{key}: {value}" for key, value in config.items()])
    logger.info(f"Simulation Configuration:\n{formatted_config}")


def log_simulation_start():
    logger = logging.getLogger()
    logger.info("------ Start of New Simulation: %s ------", datetime.datetime.now())


def print_to_log(*args, **kwargs):
    logger = logging.getLogger()
    # logger.info(" ".join(map(str, args)))
    logger.info(*args, **kwargs)
