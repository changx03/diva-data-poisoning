"""Apply Poisoning SVM Attack on feature space
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from secml.adv.attacks import CAttackPoisoningSVM

from label_flip_revised.utils import (create_dir, open_csv, open_json,
                                      time2str, to_csv, to_json)

N_ITER_SEARCH = 50  # Number of iteration for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(1e0, 1e3),
    'gamma': loguniform(1e-4, 1e2),
    'kernel': ['rbf'],
}
SOLVER_PARAMS = {
    'eta': 0.05,
    'eta_min': 0.05,
    'eta_max': None,
    'max_iter': 100,
    'eps': 1e-6
}
STEP = 0.1
TEST_SIZE = 0.2


def run_attack(X, y, X_val, y_val, C, gamma, r):
    clip_values = (X.min(), X.max())
    X_train = CArray(X)
    y_train = CArray(y)
    X_val = CArray(X_val)
    y_val = CArray(y_val)
    train_set = CDataset(X_train, y_train)
    val_set = CDataset(X_val, y_val)

    clf = CClassifierSVM(C=C, kernel=CKernelRBF(gamma=gamma))
    clf.fit(X_train, y_train)

    attack = CAttackPoisoningSVM(
        classifier=clf,
        training_data=train_set,
        val=val_set,
        lb=clip_values[0],
        ub=clip_values[1],
        solver_params=SOLVER_PARAMS,
    )
    # Initial poisoning sample
    xc = train_set[0, :].X
    yc = train_set[0, :].Y
    attack.x0 = xc
    attack.xc = xc
    attack.yc = yc

    # Set # of poisoning examples
    n_poison = int(np.floor(X_train.shape[0] * r))
    attack.n_points = n_poison

    # Run attack
    _, _, pois_examples, _ = attack.run(X_val, y_val)
    X_poisoned = np.vstack([X, pois_examples.X.get_data()])
    y_poisoned = np.concatenate([y, pois_examples.Y.get_data()])
    return X_poisoned, y_poisoned


def attack_data(X, y, X_test, y_test, clf, path_output_base, cols, advx_range,
                accuracy_poisoned, accuracy_test, filepath_train):
    for p in advx_range:
        time_start = time.perf_counter()
        # Generate poison data
        svm_params = clf.get_params()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_pois, y_pois = run_attack(X_train, y_train, X_val, y_val, C=svm_params['C'], gamma=svm_params['gamma'], r=p)
        time_elapse = time.perf_counter() - time_start

        # Save poisoned data
        path_output = '{}_poison_svm_{:.2f}.csv'.format(path_output_base, np.round(p, 2))

        # Evaluate and save results
        filepath_train.append(path_output)
        to_csv(X_pois, y_pois, cols, path_output)
        clf_pois = SVC(**svm_params)
        clf_pois.fit(X_pois, y_pois)
        acc_pois = clf_pois.score(X_pois, y_pois)
        acc_test = clf_pois.score(X_test, y_test)
        accuracy_poisoned.append(acc_pois)
        accuracy_test.append(acc_test)
        print(f'Time: {time2str(time_elapse)} Acc poisoned train: {acc_pois*100:.2f}% clean test: {acc_test*100:.2f}%')


def gen_poison_labels(path_data, path_output, advx_range, test_size):
    X, y, cols = open_csv(path_data, label_name='Class')

    # Remove extension
    dataname = Path(path_data).stem
    print(dataname)

    # Check best parameters for SVM
    path_svm_json = os.path.join(path_output, 'svm', dataname + '_svm.json')
    if os.path.exists(path_svm_json):
        best_params = open_json(path_svm_json)
        print(f'Found SVM params')
    else:
        clf = SVC()
        random_search = RandomizedSearchCV(
            clf,
            param_distributions=SVM_PARAM_DICT,
            n_iter=N_ITER_SEARCH,
            cv=5,
            n_jobs=-1,
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('random_search', random_search)])
        pipe.fit(X, y)
        best_params = pipe['random_search'].best_params_
        # Save SVM params as JSON
        to_json(best_params, path_svm_json)

    # Do NOT split the data, if train and test sets already exit.
    path_clean_train = os.path.join(
        path_output, 'train', dataname + '_clean_train.csv')
    path_clean_test = os.path.join(
        path_output, 'test', dataname + '_clean_test.csv')

    # Cannot find existing train and test sets:
    if (not os.path.exists(path_clean_train) or
            not os.path.exists(path_clean_test)):
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

    filepath_train = [path_clean_train]
    accuracy_poisoned = []
    accuracy_test = []

    # Train model
    clf = SVC(**best_params)
    clf.fit(X_train, y_train)

    acc_train = clf.score(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    accuracy_poisoned.append(acc_train)
    accuracy_test.append(acc_test)
    print('[{}] Acc on clean train: {:.2f} test: {:.2f}'.format(
        dataname, acc_train * 100, acc_test * 100))

    attack_data(
        X_train, y_train,
        X_test, y_test,
        clf,
        os.path.join(path_output, 'svm', dataname),
        cols,
        advx_range,
        accuracy_poisoned,
        accuracy_test,
        filepath_train,
    )
    data = {
        'PathTrain': np.array(filepath_train),
        'PathTest': np.array([path_clean_test] * len(filepath_train)),
        'Train': accuracy_poisoned,
        'Test': accuracy_test
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path_output, 'svm', f'{dataname}_poison_svm_score.csv'), index=False)


if __name__ == '__main__':
    # Example
    # python ./experiments/real/svm/step2_poison_svm.py -f ./data/standard/abalone_subset_std.csv -o ./data/output -t 0.2 -s 0.05 -m 0.41
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path')
    parser.add_argument('-s', '--step', type=float, default=STEP,
                        help='Spacing between values. Default=0.1')
    parser.add_argument('-m', '--max', type=float, default=0.49,
                        help='End of interval. Default=0.49')
    parser.add_argument('-t', '--test', type=float, default=TEST_SIZE,
                        help='Test set size.')
    args = parser.parse_args()
    filepath = Path(args.filepath).absolute()
    output = args.output
    step = args.step
    max_ = args.max
    test_size = args.test
    test_size = int(test_size) if test_size > 1. else float(test_size)

    advx_range = np.arange(0, max_, step)[1:]  # Remove 0%

    print(f'Path: {filepath}')
    print(f'Range: {advx_range}')

    # Create directory if not exist
    create_dir(os.path.join(output, 'svm'))
    create_dir(os.path.join(output, 'train'))
    create_dir(os.path.join(output, 'test'))

    gen_poison_labels(path_data=filepath,
                      path_output=output,
                      advx_range=advx_range,
                      test_size=test_size)
