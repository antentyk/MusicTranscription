import logging
import sys
import datetime

from src.config import config

def get_logger():
    """
    Creates configured logger,
    that will be used throughout the project.

    It will output information about
        - time
        - module
        - function name
        - logging level
        - logging message
    
    It will also create .log file in associated folder
    (you should specify it in config)

    Returns:
        logging.Logger: logger, described above
    """
    logger = logging.getLogger("application")

    formatter = logging.Formatter("%(asctime)s — %(module)s — %(funcName)s — %(levelname)s — %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    filename = config["log_folder"] + "/" + str(datetime.datetime.now()) + ".log"

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

logger = get_logger()
