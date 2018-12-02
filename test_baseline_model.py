import numpy as np

import torch

from src.model import Baseline, train_model_batch, round_probabilities

X = np.load('./src/preprocessing/MAPS_MUS-chpn_op25_e2_AkPnBcht_data.npy')
Y = np.load('./src/preprocessing/MAPS_MUS-chpn_op25_e2_AkPnBcht_labels.npy')

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

model = Baseline()
model = model.to(device)

train_model_batch(model, X, Y)

pred = model.forward(X)

pred = round_probabilities(pred.data.numpy())
