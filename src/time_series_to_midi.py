import os

import tqdm
import midiutil


def time_series_to_midi(time_series, output_filepath):
    """
    Generates midi file from time series representation of the song

    Args:
        time_series (pandas.DataFrame): time series representation of the song
            in MAPS database (Onset, Offset, MidiPitch)
        output_filepath (str): path to desired midi file

    Returns:
        None
    """
    output_midi = midiutil.MIDIFile(1)
    output_midi.addTrackName(0, 0, "Test track")
    output_midi.addTempo(0, 0, 120)

    for index, row in tqdm.tqdm(time_series.iterrows()):
        output_midi.addNote(
            0,
            0,
            int(row["MidiPitch"]),
            float(row["OnsetTime"]),
            float(row["OffsetTime"] - row["OnsetTime"]),
            100
        )

    with open(output_filepath, "wb") as binfile:
        output_midi.writeFile(binfile)
