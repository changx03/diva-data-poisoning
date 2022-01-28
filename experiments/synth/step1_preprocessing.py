"""Generate synthetic data.
"""
import argparse
import datetime
import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

N_SAMPLES = 2000  # 1000 for training; 1000 for testing
N_CLASSES = 2  # Number of classes
N_SETS = 150  # Number of dataset want to generate
N_DIFFICULTY = 3
DIFFICULTY_RANGE = [0.7, 0.9]
N_SETS = 1000


def save_data(df, file_name, data_path):
    path_output = os.path.join(data_path, f'{file_name}.csv')
    df.to_csv(path_output, index=False)


def gen_synth_data(data_path, param, bins):
    X, y = make_classification(**param)
    normalizer = Normalizer().fit(X)
    X = normalizer.transform(X)
    feature_names = ['x' + str(i) for i in range(1, X.shape[1] + 1)]

    # To dataframe
    df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
    df['y'] = y
    df['y'] = df['y'].astype('category')

    # Format name based on params
    file_name = 'f{:02d}_i{:02d}_r{:02d}_c{:02d}_w{:.0f}_'.format(
        param['n_features'],
        param['n_informative'],
        param['n_redundant'],
        param['n_clusters_per_class'],
        param['weights'][0] * 10)
    data_list = glob.glob(os.path.join(data_path, file_name + '*.csv'))
    file_name += str(len(data_list) + 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)
    clf = SVC()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    if acc <= DIFFICULTY_RANGE[0] and bins[0] > 0:
        save_data(df, file_name, data_path)
        bins[0] -= 1
    elif acc <= DIFFICULTY_RANGE[1] and bins[1] > 0:
        save_data(df, file_name, data_path)
        bins[1] -= 1
    elif acc > DIFFICULTY_RANGE[1] and bins[12] > 0:
        save_data(df, file_name, data_path)
        bins[2] -= 1
    else:
        print(f'Ditch {file_name}')

    path_output = os.path.join(data_path, f'{file_name}.csv')
    # Save dataframe
    df.to_csv(path_output, index=False)
    print('Save to:', path_output)


def synth_data_grid(n_sets, folder):
    n_per_bin = n_sets // N_DIFFICULTY
    bins = n_per_bin * np.ones(N_DIFFICULTY)

    # Create directory
    data_path = os.path.join('data', folder)
    if not os.path.exists(data_path):
        print('Create path:', data_path)
        path = Path(data_path)
        path.mkdir(parents=True)

    grid = []
    for f in range(4, 31):
        grid.append({
            'n_samples': [N_SAMPLES],
            'n_classes': [N_CLASSES],
            'n_features': [f],
            'n_repeated': [0],
            'n_informative': np.arange(math.ceil(f / 2), f + 1),
            'weights': np.expand_dims([0.4, 0.5, 0.6], axis=1)})
    param_sets = list(ParameterGrid(grid))
    print('# of parameter sets:', len(param_sets))
    for i in range(len(param_sets)):
        param_sets[i]['n_redundant'] = np.random.randint(
            0, high=param_sets[i]['n_features'] + 1 - param_sets[i]['n_informative'])
        param_sets[i]['n_clusters_per_class'] = np.random.randint(
            1, param_sets[i]['n_informative'])

    # Replace iff we need more sets than it has.
    replace = len(param_sets) < N_SETS
    selected_indices = np.random.choice(
        len(param_sets), N_SETS, replace=replace)
    for i in selected_indices:
        # This ensure the generator gets a new RND seed everytime
        param_sets[i]['random_state'] = np.random.randint(
            1000, np.iinfo(np.int16).max)
        gen_synth_data(data_path, param_sets[i], bins)
        if np.sum(bins) <= 0:
            print('Generation completed!')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nSets', default=N_SETS, type=int,
                        help='# of random generated synthetic data sets.')
    parser.add_argument('-f', '--folder', default='synth', type=str,
                        help='The output folder.')
    parser.add_argument('-s', '--samples', default=N_SAMPLES, type=int,
                        help='# of random samples per data set.')
    args = parser.parse_args()
    n_sets = args.nSets
    folder = args.folder
    synth_data_grid(n_sets, folder)
