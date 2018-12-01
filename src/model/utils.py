import os
import sys

import numpy as np

from config import config

def round_probabilities(output):
    return np.vectorize(lambda x: 0 if x < config["round_bound"] else 1)(output)
