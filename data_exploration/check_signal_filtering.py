from scipy import signal
from scipy.io import wavfile
from librosa import note_to_hz

import matplotlib.pyplot as plt

from src.data_transformation import remove_low_freq, remove_high_freq

file = "./data_exploration/data/" + "bach_wtc1_C_dur.wav"
sample_rate_original, samples_original = wavfile.read(file)

# sample_rate_filtered, samples_filtered = wavfile.read("./data_exploration/data/" + "back_C_dur_filtered.wav")
# sample_rate_filtered, samples_filtered = wavfile.read("./data_exploration/data/" + "back_C_dur_filtered_high.wav")

samples_original = samples_original[:, 0]

samples_left = remove_low_freq(file, note_to_hz('C4'))
samples_left = remove_high_freq(samples_left, sample_rate_original, note_to_hz('B4'))

samples_right = remove_low_freq(file, note_to_hz('C6'))
samples_right = remove_high_freq(samples_right, sample_rate_original, note_to_hz('B6'))

wavfile.write("left.wav", sample_rate_original, samples_left)
wavfile.write("right.wav", sample_rate_original, samples_right)

l = 2000
for start in [30000, 40000, 50000, 100000, 120000, 150000]:
    fig = plt.figure(1)

    plt.subplot(311)
    plt.plot(samples_original[start:start+l])

    plt.subplot(312)
    plt.plot(samples_left[start:start+l])

    plt.subplot(313)
    plt.plot(samples_right[start:start+l])

    plt.show()