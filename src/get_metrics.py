import torch

from src import config


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

    prediction = prediction.int()
    target = target.int()

    true_positive = (prediction * target).sum(dim=1).float()
    false_positive = (prediction * (target ^ 1)).sum(dim=1).float()
    false_negative = ((prediction ^ 1) * target).sum(dim=1).float()

    res = {}

    res["precision"] = (true_positive / (true_positive + false_positive))
    res["recal"] = (true_positive / (true_positive + false_negative))
    res["accuracy"] = (true_positive / (true_positive +
                                        false_positive + false_negative))
    res["f-measure"] = (2 * res["precision"] * res["recal"]
                        ) / (res["precision"] + res["recal"])

    for item in config["metrics_names"]:
        res[item] = res[item].sum().item()

    return res
