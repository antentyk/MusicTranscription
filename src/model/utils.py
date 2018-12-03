import os
import sys

import numpy as np

from src.config import config

def round_probabilities(prediction):
    """
    This function assigns class labels (0 or 1)
    based on prediction according to "round_bound" config property.

    For example, if round_bound is 0.5 then
    [0.1, 0.6, 0.7, 0.2] transforms into [0, 1, 1, 0]
    
    Args:
        prediction(torch.tensor): prediction tensor used for rounding
            the function modifies this argument
    
    Returns:
        torch.tensor(dtype=torch.int32): rounded labels
    """
    prediction[prediction < config["round_bound"]] = 0
    prediction[prediction != 0] = 1
    prediction = prediction.int()

def get_metrics(prediction, target):
    """
    Calculates different metrics that are used to
    estimate multi-label classification given batch prediction and target

    Includes precision, recal, acccuracy and f-measure

    Args:
        prediction(torch.Tensor, shape=(samples_num, classes_num), dtype=torch.int32):
            prediction tensor
        target(torch.Tensor, shape=(samples_num, classes_num), dtype=torch.int32):
            labels tensor
    
    Returns:
        dict{str: float},
            where key is a string - name of the metric
            and value - value of the given metric
    """

    true_positive = (prediction * target).sum(dim=1).float()
    false_positive = (prediction * (target ^ 1)).sum(dim=1).float()
    false_negative = ((prediction ^ 1) * target).sum(dim=1).float()

    res = {}

    res["precision"] = (true_positive / (true_positive + false_positive)).sum().item()
    res["recal"] = (true_positive / (true_positive + false_negative)).sum().item()
    res["accuracy"] = (true_positive / (true_positive + false_positive + false_negative)).sum().item()
    res["f-measure"] = (2 * res["precision"] * res["recal"]) / (res["precision"] + res["recal"])

    return res
