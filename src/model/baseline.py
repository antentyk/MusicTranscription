import os
import sys

import torch
from torch.nn import Module, Linear, Sigmoid
from torch.optim import Adam
from torch.nn.modules.loss import MSELoss
import torch.nn.modules.loss as loss

from src.config import config

class Baseline(Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.layer = Linear(config["n_frequency_bins"], config["notes_number"])
        self.non_linear = Sigmoid()

        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.criterion = MSELoss()

    def forward(self, x):
        x = self.layer(x)
        x = self.non_linear(x)
        return x
