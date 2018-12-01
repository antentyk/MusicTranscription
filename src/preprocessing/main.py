import numpy as np
import prepare

import os
import sys

sys.path.append(os.path.abspath("../"))

import time_series
from config import config


def main():
    files = []
    # Get all WAV and TXT files
    for file in os.listdir(config["path_to_MAPS"]):
        filename, file_extension = os.path.splitext(config["path_to_MAPS"] + file)

        print(file[:-4], filename, file_extension)

        if file_extension == ".wav":
            x, y = prepare.prepare(filename + file_extension, filename + ".txt", config)
            print(x, y)

            np.save(file[:-4] + '_data', x)
            np.save(file[:-4] + '_labels', y)

            break


# Transform all of them

# Store to the output directory as numpy matrix files


if __name__ == "__main__":
    # main()

    np.save('test', np.array([1.12, 2.342343242], dtype=np.float32))
