import torch
import pandas as pd
import tqdm

from src import config


def __to_time(frame_number):
    samples = float(frame_number * config["hop_length"])
    samples /= config["sr"]
    return samples


def prediction_to_time_series(prediction):
    prediction = torch.transpose(prediction, 0, 1)

    onsets = []
    offsets = []
    pitches = []

    for note_number in tqdm.tqdm(range(config["notes_number"])):
        start = -1

        for i in range(prediction.shape[1]):
            if(prediction[note_number][i] == 0):
                if(start == -1):
                    continue
                onsetTime = __to_time(start)
                offsetTime = __to_time(i + 1)

                start = -1

                if(offsetTime - onsetTime < config["note_length_theshold"]):
                    continue

                onsets.append(onsetTime)
                offsets.append(offsetTime)
                pitches.append(note_number + config["lowest_note_midi_pitch"])
            else:
                if(start == -1):
                    start = i

    onsets = pd.DataFrame(onsets)
    offsets = pd.DataFrame(offsets)
    pitches = pd.DataFrame(pitches)

    time_series = pd.DataFrame(
        columns=["OnsetTime", "OffsetTime", "MidiPitch"])
    time_series = pd.concat([onsets, offsets, pitches], axis=1, keys=[
                            "OnsetTime", "OffsetTime", "MidiPitch"])

    return time_series
