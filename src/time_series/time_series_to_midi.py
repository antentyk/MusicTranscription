import os

import midiutil

from config import config

def time_series_to_midi(dataframe, output_filepath):
    output_midi = midiutil.MIDIFile(1)
    output_midi.addTrackName(0, 0, "Test track")
    output_midi.addTempo(0,0,120)
    
    for index, row in dataframe.iterrows():
        output_midi.addNote(
            0,
            0,
            int(row["MidiPitch"]),
            row["OnsetTime"],
            row["OffsetTime"] - row["OnsetTime"],
            100
        )
    

    output_filepath = config["ouput_folder"] + output_filepath

    try:
        os.makedirs(os.path.dirname(output_filepath))
    except FileExistsError:
        pass

    with open(output_filepath, "wb") as binfile:
        output_midi.writeFile(binfile)