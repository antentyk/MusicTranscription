import torch.nn as nn

from src import config


class Dnn3Layers(nn.Module):
    def __init__(self):
        super(Dnn3Layers, self).__init__()

        self.dense1 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn1 = nn.BatchNorm1d(config["n_bins"])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.dense2 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn2 = nn.BatchNorm1d(config["n_bins"])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)

        self.dense3 = nn.Linear(config["n_bins"], config["notes_number"])
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.relu3(x)

        return x
