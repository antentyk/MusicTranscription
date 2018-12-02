import os
import sys

import torch
from torch.nn import Module, Linear
from torch.optim import Adam
from torch.nn.modules.loss import L1Loss
import torch.nn.modules.loss as loss

from src.config import config

class Baseline(Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.layer = Linear(config["n_frequency_bins"], config["notes_number"])
        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.criterion = L1Loss()

    def forward(self, x):
        x = self.layer(x)
        x = torch.sigmoid(x)
        return x
