import os
import sys

import numpy as np

from src.config import config

def round_probabilities(prediction):
    prediction[prediction < config["round_bound"]] = 0
    prediction[prediction != 0] = 1
    prediction = prediction.int()

def get_metrics(prediction, target):
    true_positive = (prediction * target).sum(dim=1).float()
    false_positive = (prediction * (target ^ 1)).sum(dim=1).float()
    false_negative = ((prediction ^ 1) * target).sum(dim=1).float()

    res = {}

    res["precision"] = (true_positive / (true_positive + false_positive)).sum().item()
    res["recal"] = (true_positive / (true_positive + false_negative)).sum().item()
    res["accuracy"] = (true_positive / (true_positive + false_positive + false_negative)).sum().item()
    res["f-measure"] = (2 * res["precision"] * res["recal"]) / (res["precision"] + res["recal"])

    return res
