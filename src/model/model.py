import os

import torch

from src.config.config import config

class Model(torch.nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()

        self.epochs_survived = 0
        self.name = name
    
    def __save(self, name):
        dirname = config["models_folder"] + os.path.dirname(name)

        os.makedirs(dirname, exist_ok=True)
        
        torch.save(self, config["models_folder"] + name)

    def save_checkpoint(self):
        return self.__save("%s/%s.pth" % (self.name, self.epochs_survived))
    
    def save_final(self):
        return self.__save("%s/final.pth" % (self.name, ))
