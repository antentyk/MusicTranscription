import torch.nn as nn

from src import config


class Dnn10Layers(nn.Module):
    def __init__(self):
        super(Dnn10Layers, self).__init__()

        self.dense1 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn1 = nn.BatchNorm1d(config["n_bins"])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)

        self.dense2 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn2 = nn.BatchNorm1d(config["n_bins"])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)

        self.dense3 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn3 = nn.BatchNorm1d(config["n_bins"])
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)

        self.dense4 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn4 = nn.BatchNorm1d(config["n_bins"])
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.1)

        self.dense5 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn5 = nn.BatchNorm1d(config["n_bins"])
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.1)

        self.dense6 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn6 = nn.BatchNorm1d(config["n_bins"])
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.1)

        self.dense7 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn7 = nn.BatchNorm1d(config["n_bins"])
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.1)

        self.dense8 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn8 = nn.BatchNorm1d(config["n_bins"])
        self.relu8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.1)

        self.dense9 = nn.Linear(config["n_bins"], config["n_bins"])
        self.bn9 = nn.BatchNorm1d(config["n_bins"])
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=0.1)

        self.dense10 = nn.Linear(config["n_bins"], config["notes_number"])
        self.relu10 = nn.ReLU()

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
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.dense4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        x = self.dense5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.dropout5(x)

        x = self.dense6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.dropout6(x)

        x = self.dense7(x)
        x = self.relu7(x)
        x = self.bn7(x)
        x = self.dropout7(x)

        x = self.dense8(x)
        x = self.relu8(x)
        x = self.bn8(x)
        x = self.dropout8(x)

        x = self.dense9(x)
        x = self.relu9(x)
        x = self.bn9(x)
        x = self.dropout9(x)

        x = self.dense10(x)
        x = self.relu10(x)

        return x
