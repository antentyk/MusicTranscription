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

        self.epochs_survived = 0

        self.fc1 = Linear(88, 88)
        self.non_linear1 = Sigmoid()
        
        self.optimizer = Adam(self.parameters(), lr=(10 ** (-4)))
        self.criterion = BCELoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear1(x)
        return x
