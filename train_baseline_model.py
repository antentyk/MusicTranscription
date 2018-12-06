import torch

from tqdm import tqdm

from src.config import config
from src.model import Baseline, round_probabilities
from src.preprocessing import train_test_split
from src.logger import get_logger

logger = get_logger()
model = Baseline()

def save():
    logger.info("Saving model")
    torch.save(model, config["model_folder"] + "BCE_final.h5")
    logger.info("Successfully saved the model")

def main():
    logger.info("Performing train_test_split")
    train_filenames, test_filenames = train_test_split()
    logger.info("Finish performing train_test_split")

    logger.info("Start training")

    for epoch in range(config["epochs_num"]):
        logger.info("Epoch %s" % (epoch + 1, ))

        for file_name in tqdm(train_filenames):
            batch_X = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_data.tensor")
            batch_y = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_labels.tensor").float()

            model.optimizer.zero_grad()

            output = model(batch_X)
            loss = model.criterion(output, batch_y)

            loss.backward()
            model.optimizer.step()
        
        logger.info("Estimating loss...")

        loss_value = 0.0

        for file_name in tqdm(train_filenames):
            batch_X = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_data.tensor")
            batch_y = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_labels.tensor").float()

            output = model(batch_X)
            loss = model.criterion(output, batch_y)

            loss_value += loss.item()
        
        logger.info("After epoch %s loss value is %s" % (epoch + 1, loss_value))
        
        logger.info("Saving in-between model")
        torch.save(model, config["model_folder"] + ("BCE_epoch_%s.h5" % (epoch + 1, )))
        logger.info("Finished saving in-between model")
        
    logger.info("Finished training")

    save()


try:
    main()
except KeyboardInterrupt:
    logger.info("Interrupted")
    save()
