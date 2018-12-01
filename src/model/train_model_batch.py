def train_model_batch(model, batch_X, batch_y):
    model.optimizer.zero_grad()
    output = model(batch_X)
    loss = model.criterion(output, batch_y)
    loss.backward()
    model.optimizer.step()
