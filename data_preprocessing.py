import os
import logging


from tqdm import tqdm


import numpy as np
import torch

import src.preprocessing
import src.config.config as config
import src.logger.logger as logger


def save_as_numpy(data, labels, filename):
    np.save(filename + '_data.npy', data)
    np.save(filename + '_labels.npy', labels)


def save_as_tensor(data, labels, filename):
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    torch.save(data, filename + '_data.tensor')
    torch.save(labels, filename + '_labels.tensor')


def save(data, labels, filename, as_tensors=False):
    # prevent the case if there is no such directory for saving file
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if not as_tensors:
        save_as_numpy(data, labels, filename)
    else:
        save_as_tensor(data, labels, filename)


def main():
    logger.info("Start preprocessing")

    files = []

    logger.info("Building list of files")

    for (dirpath, dirnames, filenames) in os.walk(config["path_to_MAPS"]):
        filenames = filter(lambda name: name.endswith(".wav"), filenames)
        filenames = map(lambda name: name[:-4], filenames)
        for filename in filenames:
            files.append(os.path.relpath(dirpath + "/" + filename, config["path_to_MAPS"]))
    
    logger.debug("Path example: %s" % (files[0],))
    
    logger.info("Performing CQT")

    for filename in tqdm(files):
        x, y = src.preprocessing.prepare(
            config["path_to_MAPS"] + "./" + filename + ".wav",
            config["path_to_MAPS"] + "./" + filename + ".txt"
        )

        save(x, y, config["path_to_processed_MAPS"] + "./" + filename, as_tensors=True)

    logger.info("Successfully finished preprocessing")


if __name__ == "__main__":
    main()
