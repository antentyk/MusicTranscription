import os
import argparse

import torch
import torch.utils.data
import numpy as np
import tqdm

from src import get_logger, Dataset, config, cqt, round_probabilities, prediction_to_time_series, time_series_to_midi

# Process command line arguments
parser = argparse.ArgumentParser(description='Music transcription. Generate midi from wav file')
parser.add_argument('--wav', '-w', type=str, help='Path to midi file', action='store',
                    default='./data_exploration/data/bach_wtc1_c_moll.wav')
parser.add_argument('--model', '-m', type=str, help='Stored model filename(inside public models directory)', action='store',
                    default='Dnn10Layers.pth')
parser.add_argument('--out', '-o', type=str, help='Path(with the filename) where to store generated midi file', action='store',
                    default='./output.midi')

args = parser.parse_args()

# Start generating midi workflow
class SongDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self._X = X

    def __len__(self):
        return self._X.size()[0]

    def __getitem__(self, index):
        return self._X[index]

logger = get_logger(file_silent=True)

logger.info("Performing cqt")
X = cqt(args.wav)
logger.info("Done")

logger.info("Loading model")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(os.path.join(config["public_models_folder"], args.model))
model = model.to(device)
model = model.eval()

X = torch.from_numpy(X).float()

trainDatasetMeans = torch.load(os.path.join(config["public_models_folder"], "train_means.pth"))
X -= trainDatasetMeans

logger.info("Splitting into batches")

dataloader = torch.utils.data.DataLoader(SongDataset(
    X), batch_size=config["mini_batch_size"], shuffle=False)
prediction = None

logger.info("Predicting")
for batchX in tqdm.tqdm(dataloader):
    batchX = batchX.to(device)
    batchPrediction = round_probabilities(model(batchX))

    if (prediction is None):
        prediction = batchPrediction
    else:
        prediction = torch.cat((prediction, batchPrediction))

logger.info("Generating time series")
ts = prediction_to_time_series(prediction)

logger.info("Generating midi")
time_series_to_midi(ts, args.out)
