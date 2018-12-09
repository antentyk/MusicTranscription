import random
import os

import torch
import numpy as np
from tqdm import tqdm

from src.config import config
from src.logger import get_logger
from src.cqt import cqt

logger = get_logger(file_silent=True)

logger.info("Collecting list of files in the dataset")

mus_filenames = []
trills_filenames = []
scales_filenames = []
single_filenames = []
chords_filenames = []

overall_files = 0

storages = [
    mus_filenames,
    trills_filenames,
    scales_filenames,
    single_filenames,
    chords_filenames
]

needles = [
    ["MUS"],
    ["TR"],
    ["CH"],
    ["ST", "LG", "RE", "NO"],
    ["RAND", "UCHO"]
]

folders = [
    "MUS",
    "TRILLS",
    "SCALES",
    "SINGLE",
    "CHORDS"
]

for dirpath, _, filenames in tqdm(os.walk(config["path_to_MAPS"])):
    filenames = filter(lambda x: x.endswith(".wav"), filenames)
    filenames = map(lambda x: x[:-4], filenames)

    filenames = list(filenames)
    overall_files += len(filenames)

    for find_items, storage in zip(needles, storages):
        for find_item in find_items:
            storage.extend(
                map(
                    lambda x: os.path.join(dirpath, x),
                    filter(
                        lambda x: x.find("_" + find_item) != -1,
                        filenames
                    )
                )
            )

logger.info("Done!")
logger.info("Overall files: %s" % (overall_files, ))
for folder, storage in zip(folders, storages):
    logger.info("%s files: %s" % (folder, len(storage)))

logger.info("Creating folders")
for folder in folders:
    os.makedirs(os.path.join(config["path_to_processed_MAPS"], folder), exist_ok=True)
logger.info("Done!")

logger.info("Shuffling")
r = random.Random(47)
for storage in storages:
    r.shuffle(storage)
logger.info("Done!")

logger.info("Performing cqt")

for folder, filenames in zip(folders, storages):
    logger.info(folder)

    X = np.array([])
    y = np.array([])

    cnt = 0

    for filename in tqdm(filenames):
        tx, ty = cqt(filename + ".wav", filename + ".txt")

        X = np.vstack([X, tx]) if len(X) > 0 else tx
        y = np.vstack([y, ty]) if len(y) > 0 else ty

        while(X.shape[0] >= config["frames_in_file"]):
            X_tensor = torch.from_numpy(X[:config["frames_in_file"]])
            y_tensor = torch.from_numpy(y[:config["frames_in_file"]])

            torch.save(
                X_tensor,
                os.path.join(
                    config["path_to_processed_MAPS"],
                    folder,
                    ("X%s.tensor" % (cnt, ))
                )
            )
            torch.save(
                y_tensor,
                os.path.join(
                    config["path_to_processed_MAPS"],
                    folder,
                    ("y%s.tensor" % (cnt, ))
                )
            )

            cnt += 1

            X = X[config["frames_in_file"]:]
            y = y[config["frames_in_file"]:]

logger.info("Done")