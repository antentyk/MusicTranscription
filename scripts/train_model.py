import os

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.modules.loss import BCELoss

from src.config import config
from src.dataset import Dataset
from src.model.Dnn import Dnn
from src.logger import get_logger

def get_mean():
    folders = ["CHORDS", "MUS", "SCALES", "SINGLE", "TRILLS"]
    
    summ = torch.zeros(config["bins_per_note"] * 88)
    n = 0
    
    for folder in folders:
        summ += torch.load(os.path.join(config["path_to_processed_MAPS"], folder + "_sum.tensor"))
        n += torch.load(os.path.join(config["path_to_processed_MAPS"], folder + "_cnt.tensor")).item()
    
    summ /= n
    summ = summ[config["bins_per_note"] * config["lower_notes_dropout_number"]:]

    return summ

mean = get_mean()
ratio = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = get_logger(file_silent=True)

def main(
    model,
    device,
    logger,
    dataset_folder,
    optimizer,
    criterion
):
    global config
    global mean
    global ratio

    logger.info("Start training %s" % (model.name, ))
    logger.info("Dataset: %s" % (dataset_folder, ))

    for epoch in range(1, config["epochs_num"] + 1):
        logger.info("Epoch %s" % (epoch, ))
        
        dataset = Dataset(dataset_folder, "train", ratio=ratio)
        dataloader = DataLoader(dataset, batch_size=config["mini_batch_size"])

        for batch_X, batch_y in tqdm(dataloader):
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

        if(epoch % 5 == 0):
            logger.info("Estimating loss...")
            with torch.no_grad():
                for dataset_type in ["train", "validation"]:
                    loss_value = 0.0
                    
                    dataset = Dataset(dataset_folder, dataset_type, ratio=ratio)
                    dataloader = DataLoader(dataset, batch_size=config["mini_batch_size"])

                    logger.info(dataset_type)
                    logger.info("Samples num: %s" % (len(dataloader), ))

                    for batch_X, batch_y in tqdm(dataloader):
                        batch_X -= mean
                        batch_X[batch_X < 0] = 0

                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        output = model(batch_X)
                        loss = criterion(output, batch_y.float())
                        loss_value += loss.item()
                    
                    logger.info("Loss value per dataset: %s" % (loss_value, ))

model = Dnn("Dnn 50 layers 0.1 dropout SINGLE", 50, 0.1)
model = model.to(device)

main(model, device, logger, "SINGLE", Adam(model.parameters(), lr=1e-5), BCELoss())

model.name = "Dnn 50 layers 0.1 dropout SINGLE then SCALES"
model.epochs_survived = 0

main(model, device, logger, "SCALES", Adam(model.parameters(), lr=1e-5), BCELoss())

model.name = "Dnn 50 layers 0.1 dropout SINGLE then SCALES then CHORDS"
model.epochs_survived = 0

main(model, device, logger, "CHORDS", Adam(model.parameters(), lr=1e-5), BCELoss())
