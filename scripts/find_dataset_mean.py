import os

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.dataset import Dataset
from src.logger import get_logger
from src.config import config

logger = get_logger(file_silent=True)

folders = ["MUS", "TRILLS", "SCALES", "SINGLE", "CHORDS"]

sum_X = torch.zeros(config["n_bins"])
cnt = 0

for folder in folders:
    for dataset_type in ["train", "validation", "test"]:
        logger.info("Calculating %s %s" % (folder, dataset_type))

        dataset = Dataset(folder, dataset_type)
        dataloader = DataLoader(dataset)

        for batch_X, batch_y in tqdm(dataloader):
            batch_X = batch_X[0]
            batch_y = batch_y[0]
            
            if((batch_y > 0).sum().item() == 0):
                continue
            
            cnt += 1
            sum_X += batch_X

logger.info("Done")

logger.info("Saving mean")
sum_X = sum_X / cnt
torch.save(sum_X, os.path.join(config["path_to_processed_MAPS"], "dataset_mean.tensor"))
logger.info("Done")
