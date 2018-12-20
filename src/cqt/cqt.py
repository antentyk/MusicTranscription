import torch

import librosa
from scipy.io import wavfile

import numpy as np
import pandas as pd

from src.config import config


def cqt(path_to_wav, path_to_txt=None):
    sample_rate, samples = wavfile.read(path_to_wav)

    left_channel = samples[:, 0].astype(np.double)
    right_channel = samples[:, 1].astype(np.double)

    cqt = np.abs(
        librosa.core.cqt(
            left_channel,
            sr=config["sr"],
            hop_length=config["hop_length"],
            fmin=librosa.note_to_hz("C1"),
            n_bins=config["n_bins"],
            bins_per_octave=config["bins_per_octave"]
        )
    ).T
    cqt += np.abs(
        librosa.core.cqt(
            right_channel,
            sr=config["sr"],
            hop_length=config["hop_length"],
            fmin=librosa.note_to_hz(config["lowest_note"]),
            n_bins=config["n_bins"],
            bins_per_octave=config["bins_per_octave"]
        )
    ).T

    cqt = torch.from_numpy(cqt).float()

    if(path_to_txt is None):
        return cqt

    df = pd.read_csv(path_to_txt, sep="\t")

    labels = np.zeros((cqt.shape[0], config["notes_number"]), dtype=np.uint8)

    df["OnsetTime"] = np.ceil((df["OnsetTime"] * sample_rate) / config["hop_length"])
    df["OffsetTime"] = np.floor((df["OffsetTime"] * sample_rate) / config["hop_length"])

    df["OnsetTime"] = df["OnsetTime"].astype(int)
    df["OffsetTime"] = df["OffsetTime"].astype(int)

    for index, row in df.iterrows():
        startFrame = row["OnsetTime"]

        endFrame = row["OffsetTime"]

        note = row["MidiPitch"] - config["lowest_note_midi_pitch"]

        if(note < 0):
            continue

        labels[startFrame:endFrame + 1, note] = 1

    labels = torch.from_numpy(labels).int()

    return cqt, labels
