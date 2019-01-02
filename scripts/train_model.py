import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss

from src.config import config
from src.dataset import Dataset
from src.model.Dnn import Dnn3Layers
from src.logger import get_logger

folders = ["CHORDS", "MUS", "SCALES", "SINGLE", "TRILLS"]


def get_mean():
    global folders

    summ = torch.zeros(config["n_bins"])
    n = 0

    for folder in folders:
        summ += torch.load(os.path.join(
            config["path_to_processed_MAPS"], folder + "_sum.tensor"))
        n += torch.load(os.path.join(
            config["path_to_processed_MAPS"], folder + "_cnt.tensor")).item()

    summ /= n

    return summ


mean = get_mean()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = get_logger(file_silent=True)

dataloaders = [DataLoader(Dataset(folder, "train", ratio=1),
                          batch_size=config["mini_batch_size"]) for folder in folders]
dataloadersLength = [len(dataloader) for dataloader in dataloaders]

model = Dnn3Layers("Dnn 3 layers")
model = model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()

logger.info("Start training")

for epoch in range(1, config["epochs_num"] + 1):
    logger.info("Epoch %s" % (epoch, ))

    for i in tqdm(range(max(dataloadersLength))):
        for dataloader in dataloaders:
            if(len(dataloader <= i)):
                continue
            batch_X, batch_y = dataloader[i]
            batch_X -= mean
            batch_X[batch_X < 0] = 0
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.float())
            loss.backward()
            optimizer.step()

    model.epochs_survived += 1

    logger.info("Saving inbetween model...")
    model.save_checkpoint()
    logger.info("Done")

logger.info("Finished training")
