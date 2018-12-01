import librosa
from scipy.io import wavfile
import numpy as np

import os
import sys

sys.path.append(os.path.abspath("../"))

import time_series
from config import config


def q_transform(path_to_wav, config):
    """
    Performs Constant-Q Transform for wav audio file

    Args:
        path_to_wav (string): Full path to the wav file
        config(dict): Config dictionary with the following attributes:
        samples_between_cqt_columns, n_frequency_bins, bins_per_octave

    Returns:
        np.ndarray[shape=(n_bins, t), dtype=np.complex or np.float]:
        Absolute constant-Q value each frequency at each time.
        sample_rate(int): Sample rate of the wav file

    """
    _, wav_stereo = wavfile.read(path_to_wav)
    wav_samples = np.mean(wav_stereo, axis=1)

    desire_sample_rate = config["sample_rate"]
    hop_length_in = config["samples_between_cqt_columns"]
    n_bins_in = config["n_frequency_bins"]
    bins_octaves_in = config["bins_per_octave"]

    cqt = np.abs(librosa.cqt(
        wav_samples, desire_sample_rate , hop_length=hop_length_in, n_bins=n_bins_in, bins_per_octave=bins_octaves_in
    )).transpose()

    return cqt, desire_sample_rate


def get_labels(sample_rate, number_frames, path_to_txt, config):
    """
    Generate labels by using txt file with time ranges and notes

    Args:
        sample_rate(int): WAV sample rate
        number_frames(int): Number of frames
        path_to_txt (string): Full path to the midi txt file
        config(dict): Config dictionary with the following attributes:
        samples_between_cqt_columns, n_frequency_bins, bins_per_octave

    Returns:
        np.ndarray[shape=(number_frames, number_notes), dtype=int]: Labels
    """
    win_len = 512 / float(sample_rate)

    # Aux_Vector of times
    vector_aux = np.arange(1, number_frames + 1) * win_len

    # Binary labels - we need multiple labels at the same time to represent the chords
    labels = np.zeros((number_frames, config["notes_number"]))

    midi_text = time_series.time_series_from_txt(path_to_txt)

    for index, row in midi_text.iterrows():
        init_range, fin_range, pitch = float(row["OnsetTime"]), float(row["OffsetTime"]), int(row["MidiPitch"])

        # Pitch move to 0-87 range
        pitch = pitch - 21

        # Get the range indexes
        index_min = np.where(vector_aux >= init_range)
        index_max = np.where(vector_aux - 0.01 > int(fin_range * 100) / float(100))

        # set the label
        labels[index_min[0][0]:index_max[0][0], pitch] = 1

    return labels


def prepare(path_to_wav, path_to_txt, config):
    """
    Prepare single wav file for training

    Args:
        sample_rate(int): WAV sample rate
        number_frames(int): Number of frames
        path_to_txt (string): Full path to the midi txt file
        config(dict): Config dictionary with the following attributes:
        samples_between_cqt_columns, n_frequency_bins, bins_per_octave

    Returns:
        np.ndarray[shape=(number_frames, number_notes), dtype=int]: Labels
    :param path_to_wav:
    :param path_to_txt:
    :param config:
    :return:
    """
    cqt, sample_rate = q_transform(path_to_wav, config)
    number_frames = cqt.shape[0]

    labels = get_labels(sample_rate, number_frames, path_to_txt, config)

    return cqt, labels


if __name__ == "__main__":
    prepare(config["path_to_MAPS"] + "MAPS_MUS-chpn_op25_e2_AkPnBcht.wav",
            config["path_to_MAPS"] + "MAPS_MUS-chpn_op25_e2_AkPnBcht.txt",
            config)

    # qtr, sample_rate = q_transform()
    # Easy plot
    # import matplotlib.pyplot as plt
    # plt.pcolormesh(qtr)
    # plt.show()

    # Librose plot
    # import librosa.display
    # librosa.display.specshow(
    #     librosa.amplitude_to_db(qtr, ref=np.max), sr=sample_rate, x_axis='time', y_axis='cqt_note'
    # )
    #
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrum')
    # plt.tight_layout()
    #
    # plt.show()
