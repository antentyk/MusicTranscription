import numpy as np

import os
import sys

import torch

sys.path.append(os.path.abspath("./src"))

from model import Baseline, train_model_batch, round_probabilities

device = torch.device("cpu")

X = torch.load('./processed_maps/MAPS_MUS-alb_se3_AkPnBcht_data.tensor')
Y = torch.load('./processed_maps/MAPS_MUS-alb_se3_AkPnBcht_labels.tensor')

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')

model = Baseline()
model = model.to(device)

train_model_batch(model, X, Y)

pred = model.forward(X)

pred = round_probabilities(pred.data.numpy())

np.save("prediction", pred)
