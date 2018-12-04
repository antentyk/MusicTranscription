import torch

from tqdm import tqdm

from src.config import config
from src.model import Baseline, train_model_batch, round_probabilities
from src.preprocessing import train_test_split
from src.logger import logger

logger.info("Performing train_test_split")

train_filenames, test_filenames = train_test_split()

logger.debug("Slicing")
train_filenames = train_filenames[:2000]

logger.info("Finish performing train_test_split")

logger.info(len(train_filenames))
logger.info(len(test_filenames))

model = Baseline()

logger.info("Start training")

for epoch in range(config["epochs_num"]):
    logger.info("Epoch %s" % (epoch, ))

    cnt = 0
    current_loss = 0.0

    for file_name in tqdm(train_filenames):
        cnt += 1
        if(cnt % 2000 == 0):
            logger.info("%s iteration per 2000 batches, loss: %s" % (cnt // 2000, current_loss))
            current_loss = 0

        batch_X = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_data.tensor")
        batch_y = torch.load(config["path_to_processed_MAPS"] + "./" + file_name + "_labels.tensor")

        current_loss += train_model_batch(model, batch_X, batch_y)

logger.info("Finish training")


logger.info("Saving model")

torch.save(model, config["model_folder"] + "./" + "baseline.h5")

logger.info("Saved successfully")
