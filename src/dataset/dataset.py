import torch
from torch.utils import data

import random
import os

from src.config import config

class Dataset(data.Dataset):
    def __init__(self, folder, dataset_type, ratio=1):  
        self.__folder = os.path.join(config["path_to_processed_MAPS"], folder)
        if(not os.path.isdir(self.__folder)):
            raise RuntimeError("No such folder %s" % (folder, ))
        
        dataset_type = dataset_type.lower()
        if(dataset_type not in ["train", "validation", "test"]):
            raise RuntimeError("No such dataset type: %s" % (dataset_type, ))
        
        if(ratio > 1 or ratio <= 0):
            raise RuntimeError("Wrong value for ratio: %s" % (ratio, ))

        files = os.listdir(self.__folder)
        files = filter(lambda name: name.endswith(".tensor"), files)
        files = map(lambda name: int(name[1:name.find('.')]), files)

        max_batch_num = max(files)

        self.__batches_filenames = list(range(max_batch_num + 1))
        random.Random(47).shuffle(self.__batches_filenames)

        self.__batches_filenames = self.__batches_filenames[:int((max_batch_num + 1) * ratio)]

        train_size = int(config["train_size"] * len(self.__batches_filenames))
        test_size = int(config["test_size"] * len(self.__batches_filenames))
        validation_size = int(config["validation_size"] * len(self.__batches_filenames))

        if(dataset_type == "train"):
            self.__batches_filenames = self.__batches_filenames[:train_size]
        elif(dataset_type == "test"):
            self.__batches_filenames = self.__batches_filenames[train_size: train_size + test_size]
        else:
            self.__batches_filenames = self.__batches_filenames[-validation_size:]
        
        self.__prev_opened_file_index = -1
        self.__prev_opened_X = None
        self.__prev_opened_y = None

    def __len__(self):
        return config["frames_in_file"] * len(self.__batches_filenames)
    
    def __getitem__(self, index):
        file_index = self.__batches_filenames[index // config["frames_in_file"]]
        sample_index = index % config["frames_in_file"]

        if(file_index != self.__prev_opened_file_index):
            self.__prev_opened_file_index = file_index
            self.__prev_opened_X = torch.load(os.path.join(self.__folder, ("X%s.tensor" % (file_index, ))))
            self.__prev_opened_y = torch.load(os.path.join(self.__folder, ("y%s.tensor" % (file_index, ))))
        
        X = self.__prev_opened_X[sample_index]
        y = self.__prev_opened_y[sample_index]

        X = X[:,config["lower_notes_dropout_number"] * config["bins_per_note"]:]
        y = y[:,config["lower_notes_dropout_number"]:]

        return X, y
