import torch
import torch.utils.data

import numpy as np

import tqdm

from src import get_logger, Dataset, config, cqt, round_probabilities, prediction_to_time_series, time_series_to_midi

logger = get_logger(file_silent=False)

logger.info("Performing cqt")
X = cqt("./data_exploration/data/MAPS_MUS-chpn-p4_AkPnBcht.wav")
logger.info("Done")

logger.info("Loading model")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/095.pth")
model = model.to(device)
model = model.eval()

p = np.random.permutation(X.shape[0])
invp = np.argsort(p)

X = X[p]

trainDataset = Dataset(config["path_to_processed_MAPS"], "train")

X = torch.from_numpy(X).float()
X -= trainDataset.mean()

logger.info("Splitting into batches")

class SongDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self._X = X

    def __len__(self):
        return self._X.size()[0]

    def __getitem__(self, index):
        return self._X[index]


dataloader = torch.utils.data.DataLoader(SongDataset(
    X), batch_size=config["mini_batch_size"], shuffle=False)
prediction = None

logger.info("Predicting")

for batchX in tqdm.tqdm(dataloader):
    batchX = batchX.to(device)
    batchPrediction = round_probabilities(model(batchX))

    if(prediction is None):
        prediction = batchPrediction
    else:
        prediction = torch.cat((prediction, batchPrediction))

prediction = prediction[invp]

logger.info("Generating time series")

ts = prediction_to_time_series(prediction)
logger.info("Generating midi")
time_series_to_midi(ts, "./test.midi")
