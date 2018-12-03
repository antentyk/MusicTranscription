def train_model_batch(model, batch_X, batch_y):
    """
    This function trains a model with a batch provided
    based on model's optimizer and loss function

    Args:
        model(torch.nn.Module): model to be updated
            It is assumed, that model has
            criterion and optimizer fields
        batch_X(torch.Tensor): data to feed the model
        batch_y(torch.Tensor): labels for the batch
    
    Returns:
        None
    """
    model.optimizer.zero_grad()
    output = model(batch_X)
    loss = model.criterion(output, batch_y)
    loss.backward()
    model.optimizer.step()
