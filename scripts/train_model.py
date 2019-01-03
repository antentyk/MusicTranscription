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

from tensorboardX import SummaryWriter

folders = ["MUS"]


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

model = Dnn3Layers("Kokoko")

# model = Dnn3Layers("Dnn 3 layers")
model = model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()

logger.info("Start training")

writer = SummaryWriter()

epochCounter = 0
counter = 0

for epoch in range(model.epochs_survived + 1, config["epochs_num"] + 1):
    logger.info("Epoch %s" % (epoch, ))

    dataloaders = [DataLoader(Dataset(
        folder, "train"), batch_size=config["mini_batch_size"]).__iter__() for folder in folders]
    maxDataloaderLength = max([len(dataloader) for dataloader in dataloaders])

    tqdmIter = tqdm(range(maxDataloaderLength))

    losses = []

    for i in tqdmIter:
        newDataloaders = []
        for dataloader in dataloaders:
            batch_X, batch_y = None, None

            try:
                batch_X, batch_y = dataloader.next()
                newDataloaders.append(dataloader)
            except StopIteration:
                continue

            batch_X -= mean
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.float())
            tqdmIter.set_description(str(loss.item()))
            writer.add_scalar("loss", loss.item(), counter)
            losses.append(loss.item())
            counter += 1
            loss.backward()
            optimizer.step()
        dataloaders = newDataloaders

    writer.add_scalar("lossPerEpoch", sum(losses) / len(losses), epochCounter)
    epochCounter += 1

    model.epochs_survived += 1

    logger.info("Saving inbetween model...")
    model.save_checkpoint()
    logger.info("Done")

    continue

    logger.info("Estimating loss...")
    loss_value = 0.0
    for folder in folders:
        dataloader = DataLoader(Dataset(folder, "train"), batch_size=config["mini_batch_size"])
        logger.info(folder)
        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader):
                batch_X -= mean
                batch_X[batch_X < 0] = 0

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                output = model(batch_X)
                loss = criterion(output, batch_y.float())
                loss_value += loss.item()
    logger.info("Loss value per dataset: %s" % (loss_value, ))

logger.info("Finished training")

