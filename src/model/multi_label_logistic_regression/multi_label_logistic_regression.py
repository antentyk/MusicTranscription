from src.model.model import Model
from torch.nn import Linear, Sigmoid, ReLU
from src.config import config


class MultiLabelLogisticRegression(Model):
    def __init__(self, name):
        super().__init__(name)

        self.fc1 = Linear(config["n_bins"], int(config["n_bins"] / 2))
        self.non_linear1 = ReLU()

        self.fc2 = Linear(int(config["n_bins"] / 2), 88)
        self.non_linear2 = Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear1(x)
        x = self.fc2(x)
        x = self.non_linear2(x)
        return x
