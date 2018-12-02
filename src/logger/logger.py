import logging
import sys

def get_logger():
    logger = logging.getLogger("application")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s — %(module)s — %(funcName)s — %(levelname)s — %(message)s"))

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger

logger = get_logger()