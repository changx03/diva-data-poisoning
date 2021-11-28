import torch
import torch.nn as nn
import torch.nn.functional as F

from .earlystopping import EarlyStopping


def train(dataloader, model, loss_fn, optimizer, device):
    n = len(dataloader.dataset)
    n_batches = len(dataloader)
    loss_avg, correct = 0, 0

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        assert X.size(0) == y.size(0)

        optimizer.zero_grad()
        output = model(X)
        assert output.size(0) == y.size(0)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        correct += (output.argmax(1) == y).type(torch.float).sum().item()

    loss_avg /= n_batches
    acc = correct / n
    return acc, loss_avg


def evaluate(dataloader, model, loss_fn, device):
    n = len(dataloader.dataset)
    n_batches = len(dataloader)
    loss_avg, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            assert output.size(0) == y.size(0)
            loss_avg += loss_fn(output, y).item()
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

    loss_avg /= n_batches
    acc = correct / n
    return acc, loss_avg


def train_model(model, dataloader, optimizer, loss_fn, device, max_epochs):
    early_stopping = EarlyStopping()

    for epoch in range(max_epochs):
        acc_train, loss_train = train(
            dataloader, model, loss_fn, optimizer, device)
        early_stopping(loss_train)
        if early_stopping.early_stop:
            # print('Stop at: {}'.format(epoch))
            break
    return acc_train, loss_train
