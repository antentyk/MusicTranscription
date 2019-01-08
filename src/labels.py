import math

import pandas as pd
import numpy as np

from src import config


def labels(path_to_txt, cqtResult):
    df = pd.read_csv(path_to_txt, sep='\t')
    labels = np.zeros((cqtResult.shape[0], config["notes_number"]), dtype=int)

    for _, row in df.iterrows():
        startTime = row["OnsetTime"]
        endTime = row["OffsetTime"]

        note = int(row["MidiPitch"]) - config["lowest_note_midi_pitch"]
        if(note < 0 or note >= config["notes_number"]):
            continue

        startFrame = math.ceil((startTime * config["sr"]) / config["hop_length"])
        endFrame = math.floor((endTime * config["sr"]) / config["hop_length"])

        if(startFrame > endFrame):
            continue

        labels[startFrame:endFrame + 1, note] = 1

    return labels
