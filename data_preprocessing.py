import os


import numpy as np
import torch

import src.preprocessing as preprocessing
import src.config.config as config


def save_as_numpy(data, labels, filename):
    np.save(config["path_to_processed_MAPS"] + filename + '_data', data)
    np.save(config["path_to_processed_MAPS"] + filename + '_labels', labels)


def save_as_tensor(data, labels, filename):
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    torch.save(data, config["path_to_processed_MAPS"] + filename + '_data.tensor')
    torch.save(labels, config["path_to_processed_MAPS"] + filename + '_labels.tensor')


def save(data, labels, filename, as_tensors=False):
    # prevent the case if there is no such directory for saving file
    os.makedirs(os.path.dirname(config["path_to_processed_MAPS"]), exist_ok=True)

    if not as_tensors:
        save_as_numpy(data, labels, filename)
    else:
        save_as_tensor(data, labels, filename)


def main():
    print("=== Start preprocessing ===")

    files = []
    # Get all WAV and TXT files
    for file in os.listdir(config["path_to_MAPS"]):
        filename, file_extension = os.path.splitext(config["path_to_MAPS"] + file)

        if file_extension == ".wav":
            print("Processing: ", file)

            # Transform to data and labels matrices
            x, y = preprocessing.prepare(filename + file_extension, filename + ".txt", config)

            # Store to the output directory as numpy matrix files
            save(x, y, file[:-4], as_tensors=True)

    print("=== Successfully finished preprocessing ===")


if __name__ == "__main__":
    main()
