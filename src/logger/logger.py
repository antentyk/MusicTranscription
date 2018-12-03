import logging
import sys

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

    Returns:
        logging.Logger: logger, described above
    """
    logger = logging.getLogger("application")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s — %(module)s — %(funcName)s — %(levelname)s — %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger

logger = get_logger()
