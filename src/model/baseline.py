import os
import sys

import torch
from torch.nn import Module, Linear, Sigmoid, ReLU
from torch.optim import Adam
from torch.nn.modules.loss import BCELoss
import torch.nn.modules.loss as loss

from src.config import config

class Baseline(Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.fc1 = Linear(
            config["bins_per_note"] * 88,
            (config["bins_per_note"] - 1) * 88
        )
        self.non_linear1 = ReLU()

        self.fc2 = Linear(
            (config["bins_per_note"] - 1) * 88,
            88
        )
        self.non_linear2 = Sigmoid()

        self.optimizer = Adam(self.parameters(), lr=1e-3)
        self.criterion = BCELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear1(x)
        x = self.fc2(x)
        x = self.non_linear2(x)
        return x
