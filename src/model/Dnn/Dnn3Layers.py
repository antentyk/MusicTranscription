from torch.nn import Linear, Sigmoid, ReLU, Dropout, Sequential

import numpy as np

from src.model import Model
from src.config import config


class Dnn3Layers(Model):
    def __init__(self):
        super().__init__(name)

        tmp = []
        for i in range(3):
            tmp.append(Linear(config["n_bins"], config["n_bins"]))
            tmp.append(ReLU())
            tmp.append(Dropout(0.2))
        tmp.append(Linear(config["n_bins"], config["notes_number"]))
        tmp.append(Sigmoid())

        self.layers = Sequential(*tmp)

    def forward(self, x):
        x = self.layers(x)
        return x
