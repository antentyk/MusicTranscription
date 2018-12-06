import math

import torch

import librosa
from scipy.io import wavfile

import numpy as np
import pandas as pd

from src.config import config

def prepare(path_to_wav, path_to_txt):
    sample_rate, samples = wavfile.read(path_to_wav)
    samples = np.mean(samples, axis=1)

    cqt = np.abs(
        librosa.core.cqt(
            samples,
            sr=config["sr"],
            hop_length=config["hop_length"],
            fmin=librosa.note_to_hz("A0"),
            n_bins=88
        )
    ).T

    df = pd.read_csv(path_to_txt, sep="\t")

    labels = np.zeros((cqt.shape[0], 88), dtype=np.uint8)

    df["OnsetTime"] = np.ceil((df["OnsetTime"] * sample_rate) / config["hop_length"])
    df["OffsetTime"] = np.floor((df["OffsetTime"] * sample_rate) / config["hop_length"])

    df["OnsetTime"] = df["OnsetTime"].astype(int)
    df["OffsetTime"] = df["OffsetTime"].astype(int)

    for index, row in df.iterrows():
        startFrame = row["OnsetTime"]

        endFrame = row["OffsetTime"]
        
        note = row["MidiPitch"] - 21
        labels[startFrame:endFrame + 1, note] = 1
    
    labels = torch.from_numpy(labels).int()
    cqt = torch.from_numpy(cqt).float()

    return cqt, labels

