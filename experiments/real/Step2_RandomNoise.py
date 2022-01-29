import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

from label_flip_revised import flip_random
from label_flip_revised.utils import (create_dir, open_csv, open_json, to_csv,
                                      to_json)

N_ITER_SEARCH = 50  # Number of iteration for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(0.01, 10),
    'gamma': loguniform(0.01, 10),
    'kernel': ['rbf'],
}


def run_random_flipping(path_train, path_test, dataname, advx_range, path_data, path_output):
    print(dataname)
    create_dir(os.path.join(path_data, 'rand_svm'))
    create_dir(os.path.join(path_output, 'rand_svm'))

    df = pd.DataFrame()

    # Load data
    X_train, y_train, cols = open_csv(path_train)
    X_test, y_test, _ = open_csv(path_test)

    path_svm_json = os.path.join(path_output, 'rand_svm', dataname + '_svm.json')
    if os.path.exists(path_svm_json):
        best_params = open_json(path_svm_json)
    else:
        # Tune parameters
        clf = SVC()
        random_search = RandomizedSearchCV(clf, param_distributions=SVM_PARAM_DICT,
                                           n_iter=N_ITER_SEARCH, cv=5, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        # best_params = BEST_PARAMS
        # Save SVM params as JSON
        to_json(best_params, path_svm_json)

    # Train model
    clf = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
    clf.fit(X_train, y_train)
    acc_train_clean = clf.score(X_train, y_train)
    acc_test_clean = clf.score(X_test, y_test)

    accuracy_train_clean = [acc_train_clean] * len(advx_range)
    accuracy_test_clean = [acc_test_clean] * len(advx_range)
    accuracy_train_poison = []
    accuracy_test_poison = []
    path_poison_data_list = []

    for p in advx_range:
        path_poison_data = os.path.join(path_data, 'rand_svm', f'{dataname}_rand_svm_{p:.2f}.csv')
        try:
            if os.path.exists(path_poison_data):
                X_train, y_flip, _ = open_csv(path_poison_data)
            else:
                y_flip = flip_random(y_train, p)
                to_csv(X_train, y_flip, cols, path_poison_data)

            svm_params = clf.get_params()
            clf_poison = SVC(**svm_params)
            clf_poison.fit(X_train, y_flip)
            acc_train_poison = clf_poison.score(X_train, y_flip)
            acc_test_poison = clf_poison.score(X_test, y_test)

        except Exception as e:
            print(e)
            acc_train_poison = 0
            acc_test_poison = 0
        print('P-Rate {:.2f} Acc  P-train: {:.2f} C-test: {:.2f}'.format(
            p * 100, acc_train_poison * 100, acc_test_poison * 100))
        path_poison_data_list.append(path_poison_data)
        accuracy_train_poison.append(acc_train_poison)
        accuracy_test_poison.append(acc_test_poison)

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
    df_ = pd.DataFrame(data)
    df = pd.concat([df, df_])
    df.to_csv(os.path.join(path_output, f'{dataname}_rand_svm_score.csv'), index=False)


if __name__ == '__main__':
    # Example:
    # python ./experiments/real/Step2_RandomNoise.py -f data/real -d "breastcancer_std"
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

    run_random_flipping(path_train, path_test, dataset, advx_range, filepath, output)
