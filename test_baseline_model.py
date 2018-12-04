import torch

from src.logger import logger
from src.config import config
from src.preprocessing import train_test_split
from src.model import round_probabilities
from src.time_series import prediction_to_time_series, time_series_to_midi

model = torch.load(config["model_folder"] + "baseline.h5")

song_path = "/..."

X = torch.load(config["path_to_processed_MAPS"] + song_path + "_data.torch")

pred = round_probabilities(model(X))
pred = prediction_to_time_series(pred, config["sample_rate"])
pred = time_series_to_midi(pred, "./", "kokoko.midi")
