import sys
import os

import torch
import numpy as np
import tqdm

from src import config, get_logger, cqt, labels

logger = get_logger(file_silent=True)

workers_num = int(sys.argv[1])
current_worker_num = int(sys.argv[2])

folders = sorted(os.listdir(config["path_to_MAPS"]))
folders = filter(lambda item: os.path.isdir(
    os.path.join(config["path_to_MAPS"], item)), folders)
folders = list(folders)[current_worker_num::workers_num]

for folder in folders:
    logger.info("Folder %s" % (folder, ))

    os.mkdir(os.path.join(config["path_to_processed_MAPS"], folder))

    logger.info("Searching files")
    allFiles = []

    for dirpath, _, filenames in os.walk(os.path.join(config["path_to_MAPS"], folder)):
        filenames = filter(
            lambda filename: filename.endswith(".wav"), filenames)
        filenames = map(lambda filename: filename[:-4], filenames)
        filenames = map(lambda filename: os.path.join(
            dirpath, filename), filenames)

        allFiles.extend(filenames)

    logger.info("Performing cqt")

    X, y = None, None

    for filename in tqdm.tqdm(allFiles):
        tmp_X = cqt(filename + ".wav")
        tmp_y = labels(filename + ".txt", tmp_X)

        if(X is None):
            X = tmp_X
            y = tmp_y
        else:
            X = np.vstack((X, tmp_X))
            y = np.vstack((y, tmp_y))

    mask = (y.sum(axis=1) > 0)

    notMask = np.logical_not(mask)
    silenceSamplesNum = max(int(X.shape[0] * 0.005), notMask.sum())

    X = np.vstack((X[mask], X[notMask][:silenceSamplesNum]))
    y = np.vstack((y[mask], y[notMask][:silenceSamplesNum]))

    p = np.random.permutation(X.shape[0])
    X = X[p]
    y = y[p]

    all_Size = X.shape[0]

    train_Size = int(all_Size * config["train_ratio"])
    test_Size = int(all_Size * config["test_ratio"])
    validation_Size = int(all_Size * config["validation_ratio"])

    XTrain = torch.from_numpy(X[:train_Size])
    yTrain = torch.from_numpy(y[:train_Size])

    XTest = torch.from_numpy(X[train_Size:train_Size + test_Size])
    yTest = torch.from_numpy(y[train_Size:train_Size + test_Size])

    XValidation = torch.from_numpy(X[-validation_Size:])
    yValidation = torch.from_numpy(y[-validation_Size:])

    logger.info("Saving")

    torch.save(XTrain, os.path.join(
        config["path_to_processed_MAPS"], folder, "XTrain.tensor"))
    torch.save(yTrain, os.path.join(
        config["path_to_processed_MAPS"], folder, "yTrain.tensor"))

    torch.save(XTest, os.path.join(
        config["path_to_processed_MAPS"], folder, "XTest.tensor"))
    torch.save(yTest, os.path.join(
        config["path_to_processed_MAPS"], folder, "yTest.tensor"))

    torch.save(XValidation, os.path.join(
        config["path_to_processed_MAPS"], folder, "XValidation.tensor"))
    torch.save(yValidation, os.path.join(
        config["path_to_processed_MAPS"], folder, "yValidation.tensor"))
