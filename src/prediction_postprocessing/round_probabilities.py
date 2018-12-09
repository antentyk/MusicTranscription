from src.config.config import config

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

    return prediction
