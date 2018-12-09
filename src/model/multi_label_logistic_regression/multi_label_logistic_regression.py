import torch

from src.model.model import Model

class MultiLabelLogisticRegression(Model):
    def __init__(self, name):
        super().__init__(name)
    
        self.fc1 = torch.nn.Linear(88, 88)
        self.non_linear1 = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linear1(x)
        return x
