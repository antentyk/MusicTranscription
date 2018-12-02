import numpy as np

import os
import sys

sys.path.append(os.path.abspath("./src"))

import preprocessing
from config import config


def save(data, labels, filename):
    # prevent the case if there is no such directory for saving file
    os.makedirs(os.path.dirname(config["path_to_processed_MAPS"]), exist_ok=True)

    np.save(config["path_to_processed_MAPS"] + filename + '_data', data)
    np.save(config["path_to_processed_MAPS"] + filename + '_labels', labels)


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
            save(x, y, file[:-4])

    print("=== Successfully finished preprocessing ===")


if __name__ == "__main__":
    main()
