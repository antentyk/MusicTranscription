from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz

import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def remove_high_freq(infile_name, cut_off_freq):
    sample_rate_original, samples_original = wavfile.read(infile_name)
    samples_original = samples_original[:, 0]

    filtered_samples = butter_lowpass_filter(samples_original, cut_off_freq, sample_rate_original, 6)

    return filtered_samples


file = "./data_exploration/data/" + "bach_wtc1_C_dur.wav"
filtered_file = "./data_exploration/data/" + "back_C_dur_filtered_high.wav"

l = 2000
for start in [30000, 40000, 50000, 100000, 120000, 150000]:
    fig = plt.figure(1)

    plt.subplot(211)
    plt.plot(samples_original[start:start + l])

    plt.subplot(212)
    plt.plot(filtered_samples[start:start + l])

    plt.show()

# wavfile.write(filtered_file, sample_rate_original, filtered_samples)
