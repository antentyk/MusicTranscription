from mxm.midifile import MidiInFile, MidiToCode

file = "/media/andrew/b84c4d95-450e-4802-b12d-b33e25343b1b/home/andrew/MAPS/MAPS_AkPnStgb_1/AkPnStgb/ISOL/RE/" + "MAPS_ISOL_RE_F_S0_M76_AkPnStgb.mid"

midiIn = MidiInFile(MidiToCode(), file)

midiIn.read()
