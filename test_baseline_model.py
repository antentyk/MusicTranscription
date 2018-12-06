import torch

from src.logger import logger
from src.config import config
from src.preprocessing import train_test_split
from src.model import round_probabilities
from src.time_series import prediction_to_time_series, time_series_to_midi

model = torch.load(config["model_folder"] + "BCE_final.h5")

X = torch.load(config["path_to_processed_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-bach_846_AkPnBcht_data.tensor")
y = torch.load(config["path_to_processed_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-bach_846_AkPnBcht_labels.tensor")

pred = model(X)
pred = round_probabilities(pred)

time_series = prediction_to_time_series(pred)
time_series_to_midi(time_series, "./", "./kokoko.midi")
