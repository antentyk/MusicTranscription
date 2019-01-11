import os

import numpy as np

import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.modules.loss import MSELoss

from tensorboardX import SummaryWriter

from src import config, get_logger, Dataset, get_metrics, round_probabilities
from src.model import Dnn10Layers


logger = get_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Dnn10Layers()
model = model.to(device)
model = model.train()

optimizer = Adam(model.parameters(), lr=1e-3)

criterion = MSELoss(reduction="sum")

writer = SummaryWriter("./runs/Dnn10LayersLong")

logger.info("Loading train dataset")
trainDataset = Dataset(config["path_to_processed_MAPS"], "train")
logger.info("Done")

logger.info("Loading validation dataset")
validationDataset = Dataset(config["path_to_processed_MAPS"], "validation")
logger.info("Done")

trainDatasetMean = trainDataset.mean()

batchCounter = 0

for epoch in range(config["epochs_num"]):
    logger.info("Epoch %s" % (epoch, ))

    model = model.train()

    dataloader = DataLoader(
        trainDataset, batch_size=config["mini_batch_size"], shuffle=True)

    epochLossValue = 0

    logger.info("Start training")

    for batchX, batchy in tqdm.tqdm(dataloader):
        batchX -= trainDatasetMean

        batchX, batchy = batchX.to(device), batchy.to(device)

        optimizer.zero_grad()
        output = model(batchX)

        loss = criterion(output, batchy.float())

        epochLossValue += loss.item()

        writer.add_scalar("batchLoss", loss.item(), batchCounter)
        batchCounter += 1

        loss.backward()
        optimizer.step()

    epochLossMean = epochLossValue / len(dataloader)
    writer.add_scalar("epochLossMean", epochLossMean, epoch + 1)
    logger.info("EpochLossMean %s" % (epochLossMean))

    logger.info("Evaluating")

    with torch.no_grad():
        model = model.eval()

        trainMetrics = {}
        validationMetrics = {}

        for dataset, storage in zip(
            [trainDataset, validationDataset],
            [trainMetrics, validationMetrics]
        ):
            for batchX, batchy in tqdm.tqdm(DataLoader(dataset, batch_size=config["mini_batch_size"], shuffle=True)):
                batchX -= trainDatasetMean
                batchX, batchy = batchX.to(device), batchy.to(device)

                prediction = round_probabilities(model(batchX))

                metrics = get_metrics(prediction, batchy)

                for metric in config["metrics_names"]:
                    storage[metric] = storage.get(metric, 0) + metrics[metric]

        for metric in config["metrics_names"]:
            trainMetrics[metric] /= len(trainDataset)
            validationMetrics[metric] /= len(validationDataset)

            logger.info("Train %s: %s" % (metric, trainMetrics[metric]))
            logger.info("Validation %s: %s" %
                        (metric, validationMetrics[metric]))

            writer.add_scalars(metric,
                               {
                                   "train": trainMetrics[metric],
                                   "validation": validationMetrics[metric]
                               },
                               epoch)

    torch.save(model, os.path.join(
        config["models_folder"], "Dnn10LayersLong", (str(epoch + 1).rjust(3, "0") + ".pth")))
