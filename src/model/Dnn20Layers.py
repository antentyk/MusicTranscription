import torch.nn as nn

from src import config


class Dnn20Layers(nn.Module):
    def __init__(self):
        super(Dnn20Layers, self).__init__()

        tmp = []
        for i in range(19):
            tmp.append(nn.Linear(config["n_bins"], config["n_bins"]))
            tmp.append(nn.BatchNorm1d(config["n_bins"]))
            tmp.append(nn.ReLU())
            tmp.append(nn.Dropout(p=0.05))

        tmp.append(nn.Linear(config["n_bins"], config["notes_number"]))
        tmp.append(nn.ReLU())

        self.layers = nn.Sequential(*tmp)

    def forward(self, x):
        x = self.layers(x)
        return x
