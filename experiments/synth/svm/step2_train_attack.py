"""# Train SVM and generate Adversarial Label Flip Attack examples.
"""
import argparse
import glob
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

from label_flip_revised import alfa
from label_flip_revised.utils import (create_dir, open_csv, open_json, to_csv,
                                      to_json, transform_label)

# Ignore warnings from optimization.
# The optimization step in ALFA may fail. The attack still works despite the optimization failed.
warnings.filterwarnings('ignore')

ALFA_MAX_ITER = 5  # Number of iteration for ALFA.
N_ITER_SEARCH = 20  # Number of iteration for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(1e0, 1e3),
    'gamma': loguniform(1e-4, 0.1),
    'kernel': ['rbf'],
}
STEP = 0.1  # Increment by every STEP value.
# When a dataset has 2000 datapoints, 1000 for training, and 1000 for testing.
TEST_SIZE = 1000


def get_y_flip(X_train, y_train, p, svc):
    # Transform labels from {0, 1} to {-1, 1}
    y_train = transform_label(y_train, target=-1)
    y_flip = alfa(X_train, y_train,
                  budget=p,
                  svc_params=svc.get_params(),
                  max_iter=ALFA_MAX_ITER)
    # Transform label back to {0, 1}
    y_flip = transform_label(y_flip, target=0)
    return y_flip


def compute_and_save_flipped_data(X, y, clf, path_output_base, cols, advx_range):
    for p in advx_range:
        time_start = time.time()
        y_flip = get_y_flip(X, y, p, clf)
        time_elapse = time.time() - time_start
        print('Generating {:.0f}% poison labels took {:.1f}s'.format(
            p * 100, time_elapse))

        path_output = '{}_rbf_ALFA_{:.2f}.csv'.format(
            path_output_base, np.round(p, 2))
        to_csv(X, y_flip, cols, path_output)


def eval_outputs(path_file, dataname):
    path_data_test = os.path.join(
        path_file, 'test', dataname + '_clean_test.csv')
    X_test, y_test, _ = open_csv(path_data_test)

    # Load SVM parameters from a JSON file
    path_json_param = os.path.join(path_file, 'alfa', dataname + '_svm.json')
    svm_param = open_json(path_json_param)

    for p in advx_range:
        path_data_train = os.path.join(
            path_file, 'alfa', dataname + '_rbf_ALFA_{:.2f}.csv'.format(np.round(p, 2)))
        X_train, y_train, _ = open_csv(path_data_train)
        clf = SVC(**svm_param)
        clf.fit(X_train, y_train)
        acc_train = clf.score(X_train, y_train)
        acc_test = clf.score(X_test, y_test)
        print('Accuracy on {:.2f}% poison data train: {:.2f} test: {:.2f}'.format(
            p * 100, acc_train * 100, acc_test * 100))


def gen_poison_labels(file_list, advx_range, path_file, test_size):
    for path_data in file_list:
        # Load data
        X, y, cols = open_csv(path_data)

        # Remove extension
        dataname = os.path.splitext(os.path.basename(path_data))[0]

        # Tune parameters
        clf = SVC()
        random_search = RandomizedSearchCV(clf, param_distributions=SVM_PARAM_DICT,
                                           n_iter=N_ITER_SEARCH, cv=5, n_jobs=-1)
        random_search.fit(X, y)
        best_params = random_search.best_params_

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y)
        clf = SVC(**best_params)
        clf.fit(X_train, y_train)
        print(f'Train shape: {X_train.shape}, test shape: {X_test.shape}')

        acc_train = clf.score(X_train, y_train)
        acc_test = clf.score(X_test, y_test)

        print('[{}] Acc on clean train: {:.2f} test: {:.2f}'.format(
            dataname, acc_train * 100, acc_test * 100))

        # Save clean train and test data
        path_clean_train = os.path.join(
            path_file, 'train', dataname + '_clean_train.csv')
        to_csv(X_train, y_train, cols, path_clean_train)

        path_clean_test = os.path.join(
            path_file, 'test', dataname + '_clean_test.csv')
        to_csv(X_test, y_test, cols, path_clean_test)

        # Save SVM params as JSON
        path_svm_json = os.path.join(
            path_file, 'alfa', dataname + '_svm.json')
        to_json(best_params, path_svm_json)

        # Generate poison labels
        compute_and_save_flipped_data(
            X_train, y_train, clf,
            os.path.join(path_file, 'alfa', dataname),
            cols, advx_range)

        # Read CSV files and test results
        eval_outputs(path_file, dataname)


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
    args = parser.parse_args()
    path = args.path
    step = args.step
    max_ = args.max
    test_size = args.test
    test_size = int(test_size) if test_size > 1. else float(test_size)

    advx_range = np.arange(0, max_, step)[1:]  # Remove 0%

    print('Path:', path)
    print('Range:', advx_range)

    file_list = glob.glob(os.path.join(path, '*.csv'))
    file_list = np.sort(file_list)

    # For DEBUG only
    # file_list = file_list[:1]

    print('Found {} datasets'.format(len(file_list)))

    # Create directory if not exist
    create_dir(os.path.join(path, 'alfa'))
    create_dir(os.path.join(path, 'train'))
    create_dir(os.path.join(path, 'test'))

    gen_poison_labels(file_list, advx_range, path, test_size)
