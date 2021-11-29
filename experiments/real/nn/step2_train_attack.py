"""Train a Neural Network classifier and apply Adversarial Label Flip Attack.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from label_flip_revised.alfa_nn_v3 import get_dual_loss, solveLPNN
from label_flip_revised.simple_nn_model import SimpleModel
from label_flip_revised.torch_utils import evaluate, train_model
from label_flip_revised.utils import create_dir, open_csv, time2str, to_csv

# For data selection:
STEP = 0.1  # Increment by every STEP value.
# When a dataset has 2000 datapoints, 1000 for training, and 1000 for testing.
TEST_SIZE = 1000

# For training the classifier:
BATCH_SIZE = 128  # Size of mini-batch.
HIDDEN_LAYER = 128  # Number of hidden neurons in a hidden layer.
LR = 0.001  # Learning rate.
MAX_EPOCHS = 400  # Number of iteration for training.

# For generating ALFA:
ALFA_MAX_ITER = 3  # Number of iteration for ALFA.


def batch_train_attack(file_list,
                       advx_range,
                       path_file,
                       test_size,
                       max_epochs,
                       hidden_dim):
    pass


# TODO: NN classifiers for real data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path of the data')
    parser.add_argument('-s', '--step', type=float, default=STEP,
                        help='Spacing between values. Default=0.1')
    parser.add_argument('-m', '--max', type=float, default=0.51,
                        help='End of interval. Default=0.51')
    parser.add_argument('-t', '--test', type=float, default=TEST_SIZE,
                        help='Test set size.')
    parser.add_argument('-e', '--epoch', type=int, default=MAX_EPOCHS,
                        help='Maximum number of epochs.')
    parser.add_argument('--hidden', type=int, default=HIDDEN_LAYER,
                        help='Number of neurons in a hidden layer.')
    args = parser.parse_args()
    path = args.path
    step = args.step
    max_ = args.max
    test_size = args.test
    test_size = int(test_size) if test_size > 1. else float(test_size)
    max_epochs = args.epoch
    hidden_dim = args.hidden

    advx_range = np.arange(0, max_, step)[1:]  # Remove 0%

    print('Path:', path)
    print('Range:', advx_range)

    file_list = glob.glob(os.path.join(path, '*.csv'))
    file_list = np.sort(file_list)

    # For DEBUG only
    # file_list = file_list[:1]

    print('Found {} datasets'.format(len(file_list)))

    # Create directory if not exist
    create_dir(os.path.join(path, 'alfa_nn'))
    create_dir(os.path.join(path, 'train'))
    create_dir(os.path.join(path, 'test'))
    create_dir(os.path.join(path, 'torch'))

    batch_train_attack(file_list=file_list,
                       advx_range=advx_range,
                       path_file=path,
                       test_size=test_size,
                       max_epochs=max_epochs,
                       hidden_dim=hidden_dim)
