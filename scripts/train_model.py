from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.modules.loss import BCELoss

from src.config import config
from src.dataset import Dataset
from src.model.multi_label_logistic_regression import MultiLabelLogisticRegression
from src.logger import get_logger

dataset_folders = ["SINGLE", "CHORDS"]

logger = get_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MultiLabelLogisticRegression("all_dataset_128_hop_length").to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = BCELoss()


def main():
    logger.info("Start training")

    for epoch in range(model.epochs_survived + 1, config["epochs_num"] + 1):
        logger.info("Epoch %s" % (epoch,))

        for folder in dataset_folders:
            dataset = Dataset(folder, "train")
            dataloader = DataLoader(dataset, batch_size=512)

            for batch_X, batch_y in tqdm(dataloader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()

                output = model(batch_X)
                loss = criterion(output, batch_y.float())
                loss.backward()
                optimizer.step()

        model.epochs_survived += 1

        logger.info("Estimating loss...")

        loss_value = 0.0

        for folder in dataset_folders:
            dataset = Dataset(folder, "test")
            dataloader = DataLoader(dataset, batch_size=512)

            for batch_X, batch_y in tqdm(dataloader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                output = model(batch_X)
                loss = criterion(output, batch_y.float())
                loss_value += loss.item()

        logger.info("After epoch %s loss value is %s" % (epoch, loss_value))

        if epoch % 2 == 0:
            logger.info("Saving in-between model")
            model.save_checkpoint()
            logger.info("Finished saving in-between model")

    logger.info("Finished training")

    model.save_final()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        model.save_checkpoint()
