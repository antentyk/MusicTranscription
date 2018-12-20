import os

import torch

from src.config import config
from src.cqt import cqt
from src.prediction_postprocessing import round_probabilities
from src.data_transformation import prediction_to_time_series, time_series_to_midi
from src.logger import get_logger

logger = get_logger(file_silent=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("performing cqt")
X, y = cqt("./data_exploration/data/hallelujah.wav", "./data_exploration/data/scale.txt")
X, y = X.to(device), y.to(device)
logger.info("Done")

model = torch.load(
    os.path.join(
        config["models_folder"],
        "Mllr 4 layers SINGLE",
        "100.pth"
    )
)
model = model.to(device)

logger.info("Predicting")
prediction = model(X)
prediction = round_probabilities(prediction)
logger.info("Done")

logger.info("Generating midi")
time_series = prediction_to_time_series(prediction)
time_series_to_midi(time_series, "./test.midi")
logger.info("Done")
