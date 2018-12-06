import os

from tqdm import tqdm

import torch

from src.config import config
from src.logger import get_logger
from src.preprocessing import prepare

def main():
    logger = get_logger(file_silent=True)

    logger.info("Start preprocessing")

    files = []

    logger.info("Building list of files")

    for (dirpath, dirnames, filenames) in os.walk(config["path_to_MAPS"]):
        filenames = filter(lambda name: name.endswith(".wav"), filenames)
        filenames = map(lambda name: name[:-4], filenames)
        for filename in filenames:
            files.append(os.path.relpath(dirpath + "/" + filename, config["path_to_MAPS"]))
    
    logger.info("Performing CQT")

    for filename in tqdm(files):
        x, y = prepare(config["path_to_MAPS"] + filename + ".wav", config["path_to_MAPS"] + filename + ".txt")

        filename = config["path_to_processed_MAPS"] + filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(x, filename + "_data.tensor")
        torch.save(y, filename + "_labels.tensor")

    logger.info("Successfully finished preprocessing")


if __name__ == "__main__":
    main()
