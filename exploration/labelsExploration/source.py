import math

import librosa
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd


# Reading wav file
sample_rate, samples = wavfile.read('bach.wav')
samples = np.mean(samples, axis=1)

hop_length = 512

# Performing cqt
cqt = np.abs(
    librosa.core.cqt(
        samples,
        sr=sample_rate,
        hop_length=512,
        fmin=librosa.note_to_hz("A0"),
        n_bins=88
    )
).T

# Reading text file
df = pd.read_csv("./bach.txt", sep="\t")

# Constructing labels from dataframe
labels = np.zeros(cqt.shape, dtype=np.uint8)
for index, row in df.iterrows():
    startFrame = (float(row["OnsetTime"]) * sample_rate) / hop_length
    startFrame = int(math.ceil(startFrame))

    endFrame = (float(row["OffsetTime"]) * sample_rate) / hop_length
    endFrame = int(math.floor(endFrame))

    note = int(row["MidiPitch"])
    labels[startFrame:endFrame + 1, note] = 1


# Plotting cqt and labels

plt.figure(1)
plt.subplot(211)
plt.pcolormesh((cqt[:500]).T)

plt.subplot(212)
plt.pcolormesh((labels[:500]).T)
plt.show()
