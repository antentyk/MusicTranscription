import os
import random

from src.config import config

def train_test_split():
    """
    Performs train_test split of MAPS database
    proportions are listed in config file

    Returns:
        tuple(list(str), list(str)):
            list of relative pathes to files (without extension) (from dataset directory)
    """

    train = []
    test = []

    for (dirpath, dirnames, filenames) in os.walk(config["path_to_MAPS"]):
        filenames = filter(lambda name: name.endswith(".wav"), filenames)
        filenames = map(lambda name: name[:-4], filenames)
        filenames = map(lambda name: os.path.relpath(dirpath + "/" + name, config["path_to_MAPS"]))
        filenames = list(filenames)

        random.shuffle(filenames)

        train_len = int(len(filenames) * config["train_size"])
        test_len = len(filenames) - train_len

        train.extend(filenames[:train_len])
        test.extend(filenames[-test_len:])

    return (train, test)
