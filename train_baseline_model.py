import torch

from tqdm import tqdm

from src.config import config
from src.model import Baseline, round_probabilities
from src.preprocessing import train_test_split
from src.logger import get_logger

logger = get_logger()
model = None
try:
    model = torch.load(config["model_folder"] + "BCE_final.h5")
except FileNotFoundError:
    model = Baseline()

def save():
    logger.info("Saving model")
    torch.save(model, config["model_folder"] + "BCE_final.h5")
    logger.info("Successfully saved the model")

def main():
    logger.info("Performing train_test_split")
    train_filenames, test_filenames = train_test_split()
    train_filenames.extend(test_filenames)
    logger.info("Finish performing train_test_split")

    logger.info("Start training")

    logger.info("Loading dataset into memory")

    dataset = []
    for file_name in train_filenames:
        try:
            batch_X = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_data.tensor")
            batch_y = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_labels.tensor").float()
            
            dataset.append((batch_X, batch_y))
        except FileNotFoundError:
            pass
    
    logger.info("Finished loading dataset into memory")

    for epoch in range(model.epochs_survived + 1, config["epochs_num"] + 1):
        logger.info("Epoch %s" % (epoch, ))

        for tmp in tqdm(dataset):
            batch_X = tmp[0]
            batch_y = tmp[1]
            model.optimizer.zero_grad()
            output = model(batch_X)
            loss = model.criterion(output, batch_y)
            loss.backward()
            model.optimizer.step()

        model.epochs_survived += 1
        
        logger.info("Estimating loss...")

        loss_value = 0.0

        for tmp in tqdm(dataset):
            batch_X = tmp[0]
            batch_y = tmp[1]
            output = model(batch_X)
            loss = model.criterion(output, batch_y)
            loss_value += loss.item()

        logger.info("After epoch %s loss value is %s" % (epoch, loss_value))

        if(epoch % 5 == 0):
            logger.info("Saving in-between model")
            torch.save(model, config["model_folder"] + ("BCE_epoch_%s.h5" % (epoch, )))
            logger.info("Finished saving in-between model")
        
    logger.info("Finished training")

    save()


try:
    main()
except KeyboardInterrupt:
    logger.info("Interrupted")
    save()
