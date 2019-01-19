import os

import tqdm
import librosa

from src import get_logger, config

logger = get_logger(file_silent=True)

logger.info("Exploring single notes for repetition")

singleFiles = {}
# midiPitch: (filename, instrumentName)

logger.info("Start walking")
for root, dirs, files in tqdm.tqdm(os.walk(config["path_to_MAPS"])):
    files = filter(lambda x: x.endswith(".wav"), files)
    files = map(lambda x: x[:-4], files)
    files = filter(lambda x: "ISOL" in x, files)
    files = filter(lambda x: "NO" in x, files)
    
    for item in files:
        infoPieces = item.split('_')

        midiPitch = int(infoPieces[5][1:])
        instrumentName = infoPieces[-1]

        if(midiPitch not in singleFiles):
            singleFiles[midiPitch] = []
        
        singleFiles[midiPitch].append((os.path.join(root, item), instrumentName))
logger.info("Finished walking")

logger.info("Exploring results")
for midiPitch in tqdm.tqdm(singleFiles):
    instruments = set([item[1] for item in singleFiles[midiPitch]])

    if(len(instrumentName) <= 1):
        continue
logger.info("Finished exploring results")

logger.info("exploring A first-line octave 440 hz")
A_MIDI_PITCH = librosa.note_to_midi("A4")

files = {}
# instrument: list of files

for item in set([item[1] for item in singleFiles[A_MIDI_PITCH]]):
    files[item] = []

for item in singleFiles[A_MIDI_PITCH]:
    files[item[1]].append(os.path.relpath(item[0], start=config["path_to_MAPS"]))

logger.info("Done")

logger.info("Forming csv")
