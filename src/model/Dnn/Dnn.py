from torch.nn import Linear, Sigmoid, ReLU, Dropout, Sequential

import numpy as np

from src.model import Model
from src.config import config

class Dnn(Model):
    def __init__(self, name, layers_num, dropout_rate):
        super().__init__(name)

        initial = config["bins_per_note"] * (config["notes_number"] - config["lower_notes_dropout_number"])
        target = config["notes_number"] - config["lower_notes_dropout_number"]

        neuron_in_layers = np.linspace(initial, target, layers_num, endpoint=True).astype(int)
        tmp = []
        for i in range(1, neuron_in_layers.shape[0]):
            tmp.append(Linear(neuron_in_layers[i - 1], neuron_in_layers[i]))
            tmp.append(Dropout(dropout_rate))
            tmp.append(ReLU())
        tmp.append(Sigmoid())
        
        self.layers = Sequential(*tmp)

    def forward(self, x):
        x = self.layers(x)
        return x
