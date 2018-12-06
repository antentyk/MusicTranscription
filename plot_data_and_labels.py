import torch
import matplotlib.pyplot as plt

from src.config import config

X = torch.load(config["path_to_processed_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-bach_846_AkPnBcht_data.tensor")
y = torch.load(config["path_to_processed_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-bach_846_AkPnBcht_labels.tensor")

X = X.detach().numpy()
y = y.detach().numpy()

X = X[:500]
y = y[:500]

plt.figure(1)

plt.subplot(211)
plt.pcolormesh(X.T)

plt.subplot(212)
plt.pcolormesh(y.T)

plt.show()