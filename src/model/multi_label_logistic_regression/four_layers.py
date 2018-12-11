from torch.nn import Linear, Sigmoid, ReLU

from src.model import Model
from src.config import config

class FourLayers(Model):
    def __init__(self, name):
        super().__init__(name)

        self.fc1 = Linear(config["n_bins"], int(config["n_bins"] * 0.9))
        self.fc2 = Linear(int(config["n_bins"] * 0.9), int(config["n_bins"] * 0.7))
        self.fc3 = Linear(int(config["n_bins"] * 0.7), int(config["n_bins"] * 0.4))
        self.fc4 = Linear(int(config["n_bins"] * 0.4), 88)

        self.non_linear1 = ReLU()
        self.non_linear2 = ReLU()
        self.non_linear3 = ReLU()
        self.non_linear4 = Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear1(x)
        x = self.fc2(x)
        x = self.non_linear2(x)
        x = self.fc3(x)
        x = self.non_linear3(x)
        x = self.fc4(x)
        x = self.non_linear4(x)
        return x
