"""Train a Neural Network classifier and apply Adversarial Label Flip Attack.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
BATCH_SIZE = 256
HIDDEN_LAYER = 128  # Number of hidden neurons in a hidden layer.
LR = 0.01  # Learning rate.
MAX_EPOCHS = 300  # Number of iteration for training.
MOMENTUM = 0.9

# For generating ALFA:
ALFA_MAX_ITER = 3  # Number of iteration for ALFA.


def numpy2dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TensorDataset(
        torch.from_numpy(X).type(torch.float32),
        torch.from_numpy(y).type(torch.int64)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


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
    y_poison = np.copy(y_train).astype(int)

    pbar = tqdm(range(steps), ncols=100)
    for step in pbar:
        y_poison_next, msg = solveLPNN(alpha, tau, y_true=y_train, eps=eps)
        y_poison_next = np.round(y_poison_next).astype(int)
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


def run_attack(path_train, path_test, dataname, advx_range, path_data, path_output):
    print(dataname)
    create_dir(os.path.join(path_data, 'falfa_nn'))
    create_dir(os.path.join(path_output, 'falfa_nn'))
    create_dir(os.path.join(path_data, 'torch'))

    df = pd.DataFrame()

    # Load data
    X_train, y_train, cols = open_csv(path_train)
    X_test, y_test, _ = open_csv(path_test)

    # Step 2: Train and save the classifier
    # Prepare dataloader for PyTorch
    dataloader_train = numpy2dataloader(X_train, y_train)
    dataloader_test = numpy2dataloader(X_test, y_test, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Running on CPU!')

    n_features = X_train.shape[1]
    model = SimpleModel(n_features, hidden_dim=HIDDEN_LAYER, output_dim=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()

    path_model = os.path.join(path_data, 'torch', f'{dataname}_0.00.torch')
    if os.path.exists(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
    else:
        # Train the clean model
        time_start = time.perf_counter()
        train_model(model, dataloader_train, optimizer, loss_fn, device, MAX_EPOCHS)
        time_elapsed = time.perf_counter() - time_start
        print('Time taken: {}'.format(time2str(time_elapsed)))
        # Save model
        torch.save(model.state_dict(), path_model)

    # Evaluate results
    acc_train, _ = evaluate(dataloader_train, model, loss_fn, device)
    acc_test, _ = evaluate(dataloader_test, model, loss_fn, device)

    accuracy_train_clean = [acc_train] * len(advx_range)
    accuracy_test_clean = [acc_test] * len(advx_range)
    accuracy_train_poison = []
    accuracy_test_poison = []
    path_poison_data_list = []

    # Step 3: Generate attacks
    for p in advx_range:
        path_poison_data = os.path.join(path_data, 'falfa_nn', f'{dataname}_falfa_nn_{p:.2f}.csv')
        try:
            if os.path.exists(path_poison_data):
                X_train, y_poison, _ = open_csv(path_poison_data)
            else:
                y_poison = poison_attack(
                    model,
                    X_train,
                    y_train,
                    eps=p,
                    max_epochs=MAX_EPOCHS,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    batch_size=BATCH_SIZE,
                    device=device,
                )
                # Save attack
                to_csv(X_train, y_poison, cols, path_poison_data)
                print('Poison rate:', np.mean(y_poison != y_train))

            # Step 4: Evaluation
            dataloader_poison = numpy2dataloader(X_train, y_poison)

            # Train the poison model
            poisoned_model = SimpleModel(n_features, hidden_dim=HIDDEN_LAYER, output_dim=2).to(device)
            optimizer_poison = torch.optim.SGD(poisoned_model.parameters(), lr=LR, momentum=MOMENTUM)

            path_model = os.path.join(path_data, 'torch', f'{dataname}_{p:.2f}.torch')
            if os.path.exists(path_model):
                poisoned_model.load_state_dict(torch.load(path_model, map_location=device))
            else:
                train_model(poisoned_model, dataloader_poison, optimizer_poison, loss_fn, device, MAX_EPOCHS)
                torch.save(poisoned_model.state_dict(), path_model)

            acc_poison, _ = evaluate(dataloader_poison, poisoned_model, loss_fn, device)
            acc_test, _ = evaluate(dataloader_test, poisoned_model, loss_fn, device)
        except Exception as e:
            print(e)
            acc_poison = 0
            acc_test = 0

        print('P-Rate [{:.2f}] Acc  P-train: {:.2f} C-test: {:.2f}'.format(p * 100, acc_poison * 100, acc_test * 100))
        path_poison_data_list.append(path_poison_data)
        accuracy_train_poison.append(acc_poison)
        accuracy_test_poison.append(acc_test)
    # Save results
    data = {
        'Data': np.tile(dataname, reps=len(advx_range)),
        'Path.Train': np.tile(path_train, reps=len(advx_range)),
        'Path.Poison': path_poison_data_list,
        'Path.Test': np.tile(path_test, reps=len(advx_range)),
        'Rate': advx_range,
        'Train.Clean': accuracy_train_clean,
        'Test.Clean': accuracy_test_clean,
        'Train.Poison': accuracy_train_poison,
        'Test.Poison': accuracy_test_poison,
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path_output, f'{dataname}_falfa_nn_score.csv'), index=False)


if __name__ == '__main__':
    # Example:
    # python ./experiments/real/Step2_FALFA_NN.py -f data/real -d "breastcancer_std"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The path of the data')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('-o', '--output', type=str, default='results/real',
                        help='The output path for scores.')
    parser.add_argument('-s', '--step', type=float, default=0.05,
                        help='Spacing between values. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval. Default=0.41')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    dataset = args.dataset
    output = str(Path(args.output).absolute())
    step = args.step
    max_ = args.max

    advx_range = np.arange(0, max_, step)

    print('Path:', filepath)
    print('Range:', advx_range)

    path_train = os.path.join(filepath, 'train', f'{dataset}_train.csv')
    path_test = os.path.join(filepath, 'test', f'{dataset}_test.csv')

    run_attack(path_train, path_test, dataset, advx_range, filepath, output)
