import torch

import numpy as np
from matplotlib import pyplot as plt

from src.cqt import cqt
from src.config import config

# file = "./data_exploration/data/" + "scale"
file = "/media/andrew/b84c4d95-450e-4802-b12d-b33e25343b1b/home/andrew/MAPS/MAPS_AkPnStgb_1/AkPnStgb/ISOL/RE/" + "MAPS_ISOL_RE_F_S0_M76_AkPnStgb"

x, labels = cqt(file + ".wav", file + ".txt")

x = x.detach().numpy()
labels = labels.detach().numpy()

print("X for hop_length: %s = %s" % (config["hop_length"], x.shape))
print("Labels for hop_length: %s = %s" % (config["hop_length"], labels.shape))

for start in [0, 1000, 2000, 3000, 4000, 5000, 10000]:
    X = x[start:start + 1000]
    y = labels[start:start + 1000]

    fig = plt.figure(1)

    plt.subplot(211)
    plt.pcolormesh(X.T)

    plt.subplot(212)
    plt.pcolormesh(y.T)

    # plt.savefig('cqt%s_%s.png' % (hop_length, start))

    plt.show()
