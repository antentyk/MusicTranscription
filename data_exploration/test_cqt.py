from src.cqt import cqt
from src.config import config
import librosa.display
import numpy as np

from matplotlib import pyplot as plt

wav_file = "./data/" + "scale.wav"
txt_file = "./data/" + "scale.txt"

prediction, labels = cqt(wav_file, txt_file)

prediction = prediction.detach().numpy()
labels = labels.detach().numpy()

print("Prediction for hop_length: %s = %s" % (config["hop_length"], prediction.shape))

hop_length = config["hop_length"]

for start in [0, 1000, 2000, 3000, 4000, 5000]:
    X = prediction[start:start + 1000]
    y = labels[start:start + 1000]

    fig = plt.figure(1)

    plt.subplot(211)
    plt.pcolormesh(X.T)

    plt.subplot(212)
    plt.pcolormesh(y.T)

    # plt.savefig('cqt%s_%s.png' % (hop_length, start))

    plt.show()

