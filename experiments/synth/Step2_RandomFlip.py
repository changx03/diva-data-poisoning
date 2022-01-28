import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

from label_flip_revised import flip_random
from label_flip_revised.utils import create_dir, open_csv, to_csv, open_json, to_json

N_ITER_SEARCH = 20  # Number of iteration for SVM parameter tuning.
SVM_PARAM_DICT = {
    'C': loguniform(1e0, 1e3),
    'gamma': loguniform(1e-4, 0.1),
    'kernel': ['rbf'],
}


def gen_random_labels(train_list, test_list, advx_range, path_data, path_output):
    create_dir(os.path.join(path_data, 'rand'))
    create_dir(os.path.join(path_output, 'rand'))

    df = pd.DataFrame()
    for train, test in zip(train_list, test_list):
        dataname = Path(train).stem[: -len('_train')]
        print(dataname)
        dataname_test = Path(test).stem[: -len('_test')]
        assert dataname == dataname_test, f'{dataname} != {dataname_test}'

        # Load data
        X_train, y_train, cols = open_csv(train)
        X_test, y_test, _ = open_csv(test)

        path_svm_json = os.path.join(path_output, 'rand', dataname + '_svm.json')
        if os.path.exists(path_svm_json):
            best_params = open_json(path_svm_json)
        else:
            # Tune parameters
            clf = SVC()
            random_search = RandomizedSearchCV(
                clf,
                param_distributions=SVM_PARAM_DICT,
                n_iter=N_ITER_SEARCH,
                cv=5,
                n_jobs=-1,
            )
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            # Save SVM params as JSON
            to_json(best_params, path_svm_json)

        # Train model
        clf = SVC(**best_params)
        clf.fit(X_train, y_train)
        acc_train_clean = clf.score(X_train, y_train)
        acc_test_clean = clf.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(advx_range)
        accuracy_test_clean = [acc_test_clean] * len(advx_range)
        accuracy_train_poison = []
        accuracy_test_poison = []
        path_poison_data_list = []

        for p in advx_range:
            path_poison_data = os.path.join(path_data, 'rand', f'{dataname}_rand_{p:.2f}.csv')
            try:
                if os.path.exists(path_poison_data):
                    _, y_flip, _ = open_csv(path_poison_data)
                else:
                    if p == 0:
                        y_flip = y_train
                    else:
                        y_flip = flip_random(y_train, p)
                    to_csv(X_train, y_flip, cols, path_poison_data)

                # Evaluate
                clf_poison = SVC(**best_params)
                clf_poison.fit(X_train, y_flip)
                acc_train_poison = clf_poison.score(X_train, y_flip)
                acc_test_poison = clf_poison.score(X_test, y_test)
            except Exception as e:
                print(e)
                acc_train_poison = 0
                acc_test_poison = 0
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)
        # Save results
        data = {
            'Data': np.tile(dataname, reps=len(advx_range)),
            'Path.Train': np.tile(train, reps=len(advx_range)),
            'Path.Poison': path_poison_data_list,
            'Path.Test': np.tile(test, reps=len(advx_range)),
            'Rate': advx_range,
            'Train.Clean': accuracy_train_clean,
            'Test.Clean': accuracy_test_clean,
            'Train.Poison': accuracy_train_poison,
            'Test.Poison': accuracy_test_poison,
        }
        df_ = pd.DataFrame(data)
        df = pd.concat([df, df_])
    df.to_csv(os.path.join(path_output, 'synth_rand_svm_score.csv'), index=False)


if __name__ == '__main__':
    # Example:
    # python ./experiments/synth/Step2_RandomFlip.py -f "data/synth/"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The path of the data')
    parser.add_argument('-o', '--output', type=str, default='results/synth',
                        help='The output path for scores.')
    parser.add_argument('-s', '--step', type=float, default=0.05,
                        help='Spacing between values. Default=0.05')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval. Default=0.41')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    output = str(Path(args.output).absolute())
    step = args.step
    max_ = args.max

    advx_range = np.arange(0, max_, step)

    print('Path:', filepath)
    print('Range:', advx_range)

    train_list = sorted(glob.glob(os.path.join(filepath, 'train', '*.csv')))
    test_list = sorted(glob.glob(os.path.join(filepath, 'test', '*.csv')))
    assert len(train_list) == len(test_list)
    print('Found {} datasets'.format(len(train_list)))

    gen_random_labels(train_list, test_list, advx_range, filepath, output)
