import os

import numpy as np

import torch.utils.data

import tqdm

from src import config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, dataset_type):
        XFiles = []
        yFiles = []

        for dirpath, _, filenames in os.walk(config["path_to_processed_MAPS"]):
            filenames = map(lambda filename: os.path.join(dirpath, filename), filenames)
            filenames = filter(lambda filename: filename.endswith(".tensor"), filenames)
            
            for filename in filenames:
                if(filename.endswith("X%s.tensor" % (dataset_type.capitalize(), ))):
                    XFiles.append(filename)
                if(filename.endswith("y%s.tensor" % (dataset_type.capitalize(), ))):
                    yFiles.append(filename)

        self._X = None
        self._y = None

        progress_bar = tqdm.tqdm(range(len(XFiles)))

        for i in progress_bar:
            XFile = XFiles[i]
            yFile = yFiles[i]

            progress_bar.set_description(os.path.dirname(XFile))

            tmpX = torch.load(XFile).float()
            tmpy = torch.load(yFile).float()

            mask = ((tmpy > 0).sum(dim=1) > 0)
            tmpX = tmpX[mask]
            tmpy = tmpy[mask]

            if(tmpX.shape[0] == 0):
                continue

            if(self._X is None):
                self._X = tmpX
                self._y = tmpy
            else:
                self._X = torch.cat((self._X, tmpX))
                self._y = torch.cat((self._y, tmpy))

        p = np.random.permutation(self._X.shape[0])
        self._X = self._X[p]
        self._y = self._y[p]

    def __len__(self):
        return self._X.size()[0]

    def __getitem__(self, index):
        return self._X[index], self._y[index]

    def mean(self):
        return self._X.sum(dim=0) / len(self)
