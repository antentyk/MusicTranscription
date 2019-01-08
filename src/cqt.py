import librosa
from scipy.io import wavfile

import numpy as np

from src import config


def cqt(path_to_wav):
    _, samples = wavfile.read(path_to_wav)
    samples = samples.mean(axis=1)

    cqt = np.abs(
        librosa.core.cqt(
            samples,
            sr=config["sr"],
            hop_length=config["hop_length"],
            fmin=librosa.note_to_hz(config["lowest_note"]),
            n_bins=config["n_bins"],
            bins_per_octave=config["bins_per_octave"]
        )
    ).T

    return cqt
