import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from earlystopping import EarlyStopping
from utils import time2str


def flip_label(y, idx):
    y_flip = np.copy(y)
    y_flip[y_flip == 0] = -1
    y_flip[idx] = - y_flip[idx]
    y_flip[y_flip == -1] = 0
    return y_flip


def solveLPNN(alpha, tau, y_true, eps):
    n = len(alpha)

    c = alpha - tau

    # constraint: <=
    A_ub = np.copy(y_true).astype(float)
    A_ub[A_ub == 1] = -1
    A_ub[A_ub == 0] = 1
    A_ub = A_ub.reshape(1, -1)
    b_ub = np.dot(A_ub, y_true) + n * eps

    # Boundary
    bounds = np.array([[0., 1.]] * n)

    # The initial value must be valid.
    x0 = np.copy(y_true)
    idx = np.random.choice(n, size=int(np.floor(n * eps)), replace=False)
    x0 = flip_label(y_true, idx)

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, x0=x0,
        bounds=bounds, method='revised simplex')
    y_poison = result.x
    return y_poison, result.message


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


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


def get_dual_loss(model, X, device):
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))
        log_p = F.log_softmax(output, dim=1).cpu().detach().numpy()

    assert log_p.shape == (len(X), 2), \
        f'log_softmax expected dim: {(len(X), 2)}, received: {log_p.shape}'
    loss_dual = -log_p[:, 1] + log_p[:, 0]
    return loss_dual


def poison_attack(model,
                  X_train,
                  y_train,
                  eps,
                  max_epochs,
                  optimizer,
                  loss_fn,
                  batch_size,
                  device,
                  steps=3):
    # Compute the initial output
    X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
    tau = get_dual_loss(model, X_train_tensor, device)
    alpha = np.zeros_like(tau)
    y_poison = np.copy(y_train)

    pbar = tqdm(range(steps), ncols=100)
    for step in pbar:
        y_poison_next, msg = solveLPNN(alpha, tau, y_true=y_train, eps=eps)
        pbar.set_postfix({'Optimizer': msg})

        if step > 1 and np.all(y_poison_next == y_poison):
            print('Poison labels are converged. Break.')
            break
        y_poison = y_poison_next

        # Update model
        dataset = TensorDataset(torch.from_numpy(X_train).type(torch.float32),
                                torch.from_numpy(y_poison).type(torch.int64))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_model(model,
                    dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    max_epochs=max_epochs)
        alpha = get_dual_loss(model, X_train_tensor, device)
    return y_poison


def run_demo():
    batch_size = 128  # Mini-batch size
    lr = 0.001  # Learning rate
    max_epochs = 300  # Number of iteration for training the neural network
    eps = 0.2  # Percentage of poisoned data
    print('Epsilon:', eps)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Only select two classes
    X = X[np.where(y < 2)]
    y = y[np.where(y < 2)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare data
    dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32),
                                  torch.from_numpy(y_train).type(torch.int64))
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)

    dataset_test = TensorDataset(torch.from_numpy(X_test).type(torch.float32),
                                 torch.from_numpy(y_test).type(torch.int64))
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True)

    # Prepare models
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     torch.cuda.empty_cache()
    # else:
    #     device = torch.device('cpu')
    device = torch.device('cpu')

    model = Model(X_train.shape[1], 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    loss_fn = nn.CrossEntropyLoss()

    # Train the clean model
    time_start = time.perf_counter()
    train_model(model, dataloader_train, optimizer,
                loss_fn, device, max_epochs)
    time_elapsed = time.perf_counter() - time_start
    print('Time taken: {}'.format(time2str(time_elapsed)))

    acc_train, loss_train = evaluate(dataloader_train, model, loss_fn, device)
    acc_test, loss_test = evaluate(dataloader_test, model, loss_fn, device)
    print('[Clean] Train acc: {:.2f} loss: {:.3f}. Test acc: {:.2f} loss: {:.3f}'.format(
        acc_train * 100, loss_train, acc_test * 100, loss_test,))

    # Perform attack
    y_poison = poison_attack(model,
                             X_train,
                             y_train,
                             eps=eps,
                             max_epochs=max_epochs,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             batch_size=batch_size,
                             device=device)

    # Evaluate poison labels
    print('Poison rate:', np.mean(y_poison != y_train))

    # On Poisoned dataset
    dataset_poison = TensorDataset(
        torch.from_numpy(X_train).type(torch.float32),
        torch.from_numpy(y_poison).type(torch.int64),
    )
    dataloader_poison = DataLoader(dataset_poison, batch_size=batch_size)

    model_poison = Model(X_train.shape[1], 2).to(device)
    optimizer_poison = torch.optim.SGD(
        model_poison.parameters(), lr=lr, momentum=0.8)
    train_model(model_poison, dataloader_poison, optimizer_poison,
                loss_fn, device, max_epochs)

    acc_train, loss_train = evaluate(
        dataloader_poison, model_poison, loss_fn, device)
    acc_test, loss_test = evaluate(
        dataloader_test, model_poison, loss_fn, device)
    print('[Poisoned] Train acc: {:.2f} loss: {:.3f}. Test acc: {:.2f} loss: {:.3f}'.format(
        acc_train * 100, loss_train, acc_test * 100, loss_test,))


if __name__ == '__main__':
    run_demo()
