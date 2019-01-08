import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import cqt, labels

spectrogram = cqt("./data_exploration/data/MAPS_MUS-bach_846_AkPnBcht.wav")
ground_truth = labels(
    "./data_exploration/data/MAPS_MUS-bach_846_AkPnBcht.txt", spectrogram)

spectrogram = spectrogram[:1000]
ground_truth = ground_truth[:1000]

fig = plt.figure(1)

cmap = LinearSegmentedColormap.from_list('mapName', ['#2c3e50', '#ff5252'])

plt.subplot(211)
plt.pcolormesh(spectrogram.T, cmap=cmap)

plt.subplot(212)
plt.pcolormesh(ground_truth.T, cmap=cmap)

plt.savefig("./img/test.png")
