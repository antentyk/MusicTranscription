import logging
import sys
import datetime

from src.config import config

def get_logger(console_silent=False, file_silent=False):
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

    Args:
        console_silent(bool): whether logger should output logs in console
        file_silent(bool): whether logger should output logs in file

    Returns:
        logging.Logger: logger, described above
    """
    logger = logging.getLogger("application")

    formatter = logging.Formatter("%(asctime)s — %(module)s — %(funcName)s — %(levelname)s — %(message)s")

    if(not console_silent):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if(not file_silent):
        filename = config["log_folder"] + "/" + str(datetime.datetime.now()) + ".log"
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    logger.propagate = False

    return logger
