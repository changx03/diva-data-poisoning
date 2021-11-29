"""Train a Neural Network classifier and apply Adversarial Label Flip Attack.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from label_flip_revised.alfa_nn_v3 import get_dual_loss, solveLPNN
from label_flip_revised.simple_nn_model import SimpleModel
from label_flip_revised.torch_utils import evaluate, train_model
from label_flip_revised.utils import create_dir, open_csv, time2str, to_csv

# For data selection:
STEP = 0.05  # Increment by every STEP value.
# When a dataset has 2000 datapoints, 1000 for training, and 1000 for testing.
TEST_SIZE = 0.2

# For training the classifier:
BATCH_SIZE = 128  # Size of mini-batch.
HIDDEN_LAYER = 128  # Number of hidden neurons in a hidden layer.
LR = 0.001  # Learning rate.
MAX_EPOCHS = 400  # Number of iteration for training.

# For generating ALFA:
ALFA_MAX_ITER = 3  # Number of iteration for ALFA.


def poison_attack(model,
                  X_train,
                  y_train,
                  eps,
                  max_epochs,
                  optimizer,
                  loss_fn,
                  batch_size,
                  device,
                  steps=ALFA_MAX_ITER):
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


def batch_train_attack(path_data,
                       path_output,
                       advx_range,
                       test_size,
                       max_epochs,
                       hidden_dim):
    # Step 1: Load data
    # Remove extension
    dataname = os.path.splitext(os.path.basename(path_data))[0]

    # Do NOT split the data, if train and test sets already exit.
    path_clean_train = os.path.join(
        path_output, 'train', dataname + '_clean_train.csv')
    path_clean_test = os.path.join(
        path_output, 'test', dataname + '_clean_test.csv')

    # Cannot find train and test sets exist:
    if (not os.path.exists(path_clean_train) or
            not os.path.exists(path_clean_test)):
        X, y, cols = open_csv(path_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y)
        # Save splits.
        to_csv(X_train, y_train, cols, path_clean_train)
        to_csv(X_test, y_test, cols, path_clean_test)
    else:
        print('Found existing train-test splits.')
        X_train, y_train, cols = open_csv(path_clean_train)
        X_test, y_test, _ = open_csv(path_clean_test)

    # Preprocessing
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 2: Train and save the classifier
    # Prepare dataloader for PyTorch
    dataset_train = TensorDataset(torch.from_numpy(X_train).type(torch.float32),
                                  torch.from_numpy(y_train).type(torch.int64))
    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_test = TensorDataset(torch.from_numpy(X_test).type(torch.float32),
                                 torch.from_numpy(y_test).type(torch.int64))
    dataloader_test = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Running on CPU!')

    n_features = X_train.shape[1]
    model = SimpleModel(
        n_features, hidden_dim=hidden_dim, output_dim=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.8)
    loss_fn = nn.CrossEntropyLoss()

    # Train the clean model
    time_start = time.perf_counter()
    train_model(model, dataloader_train, optimizer,
                loss_fn, device, max_epochs)
    time_elapsed = time.perf_counter() - time_start
    print('Time taken: {}'.format(time2str(time_elapsed)))

    acc_train, loss_train = evaluate(
        dataloader_train, model, loss_fn, device)
    acc_test, loss_test = evaluate(dataloader_test, model, loss_fn, device)
    print('[Clean] Train acc: {:.2f} loss: {:.3f}. Test acc: {:.2f} loss: {:.3f}'.format(
        acc_train * 100, loss_train, acc_test * 100, loss_test,))

    # Save model
    path_model = os.path.join(
        path_output, 'torch', dataname + '_SimpleNN.torch')
    torch.save(model.state_dict(), path_model)

    # Step 3: Generate attacks
    for p in advx_range:
        y_poison = poison_attack(model,
                                 X_train,
                                 y_train,
                                 eps=p,
                                 max_epochs=max_epochs,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn,
                                 batch_size=BATCH_SIZE,
                                 device=device)
        # Save attack
        path_output = '{}_nn_ALFA_{:.2f}.csv'.format(
            os.path.join(path_output, 'alfa_nn', dataname), np.round(p, 2))
        to_csv(X_train, y_poison, cols, path_output)

        # Step 4: Evaluation
        print('Poison rate:', np.mean(y_poison != y_train))

        dataset_poison = TensorDataset(
            torch.from_numpy(X_train).type(torch.float32),
            torch.from_numpy(y_poison).type(torch.int64),
        )
        dataloader_poison = DataLoader(
            dataset_poison, batch_size=BATCH_SIZE, shuffle=True)

        # Train the poison model
        model_poison = SimpleModel(
            n_features, hidden_dim=hidden_dim, output_dim=2).to(device)
        optimizer_poison = torch.optim.SGD(
            model_poison.parameters(), lr=LR, momentum=0.8)
        train_model(model_poison, dataloader_poison, optimizer_poison,
                    loss_fn, device, max_epochs)

        acc_poison, _ = evaluate(
            dataloader_poison, model_poison, loss_fn, device)
        acc_test, _ = evaluate(
            dataloader_test, model_poison, loss_fn, device)
        print('Accuracy on {:.2f}% poison data train: {:.2f} test: {:.2f}'.format(
            p * 100, acc_poison * 100, acc_test * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path')
    parser.add_argument('-s', '--step', type=float, default=STEP,
                        help='Spacing between values. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.51,
                        help='End of interval. Default=0.51')
    parser.add_argument('-t', '--test', type=float, default=TEST_SIZE,
                        help='Test set size.')
    parser.add_argument('-e', '--epoch', type=int, default=MAX_EPOCHS,
                        help='Maximum number of epochs.')
    parser.add_argument('--hidden', type=int, default=HIDDEN_LAYER,
                        help='Number of neurons in a hidden layer.')
    args = parser.parse_args()
    filepath = Path(args.filepath).absolute()
    output = args.output
    step = args.step
    max_ = args.max
    test_size = args.test
    test_size = int(test_size) if test_size > 1. else float(test_size)
    max_epochs = args.epoch
    hidden_dim = args.hidden

    advx_range = np.arange(0, max_, step)[1:]  # Remove 0%

    print(f'Path: {filepath}')
    print(f'Range: {advx_range}')

    # Create directory if not exist
    create_dir(os.path.join(output, 'alfa_nn'))
    create_dir(os.path.join(output, 'train'))
    create_dir(os.path.join(output, 'test'))
    create_dir(os.path.join(output, 'torch'))

    batch_train_attack(path_data=filepath,
                       path_output=output,
                       advx_range=advx_range,
                       test_size=test_size,
                       max_epochs=max_epochs,
                       hidden_dim=hidden_dim)
