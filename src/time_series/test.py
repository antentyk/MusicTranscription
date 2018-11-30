import os
import sys

sys.path.append(os.path.abspath("../"))

from config import config
from time_series import *

os.chdir("../../")

df = time_series_from_txt(config["path_to_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-bach_847_AkPnBcht.txt")
time_series_to_midi(df, "test.midi")