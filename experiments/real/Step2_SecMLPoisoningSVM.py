import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from secml.adv.attacks import CAttackPoisoningSVM
from secml.array import CArray
from secml.data import CDataset
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernels import CKernelRBF
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from tqdm import tqdm

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


def run_poison_attack(path_train, path_test, dataname, advx_range, path_data, path_output):
    print(dataname)
    create_dir(os.path.join(path_data, 'poison_svm'))
    create_dir(os.path.join(path_output, 'poison_svm'))

    df = pd.DataFrame()

    # Load data
    X_train, y_train, cols = open_csv(path_train)
    X_test, y_test, _ = open_csv(path_test)

    # Preprocessing
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clip_values = (X_train.min(), X_train.max())
    print('X Range:', clip_values)

    path_svm_json = os.path.join(path_output, 'poison_svm', f'{dataname}_svm.json')
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
        # best_params = {
        #     'C': 1,
        #     'gamma': 10,
        # }
        # Save SVM params as JSON
        to_json(best_params, path_svm_json)
    print('Best params:', best_params)

    # Train model
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    clf = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
    clf.fit(X_train, y_train)
    acc_train_clean = clf.score(X_train, y_train)
    acc_test_clean = clf.score(X_test, y_test)

    accuracy_train_clean = [acc_train_clean] * len(advx_range)
    accuracy_test_clean = [acc_test_clean] * len(advx_range)
    accuracy_train_poison = []
    accuracy_test_poison = []
    path_poison_data_list = []

    # Generate poison labels
    for p in tqdm(advx_range):
        path_poison_data = os.path.join(path_data, 'poison_svm', f'{dataname}_poison_svm_{p:.2f}.csv')
        time_start = time.time()
        try:
            if os.path.exists(path_poison_data):
                X_pois, y_pois, _ = open_csv(path_poison_data)
            else:
                if p == 0:
                    X_pois = X_train
                    y_pois = y_train
                    acc_train_pois = acc_train_clean
                    acc_test_pois = acc_test_clean
                else:
                    # Using SecML wrapper
                    cX_train = CArray(X_train)
                    cy_train = CArray(y_train)
                    cX_val = CArray(X_val)
                    cy_val = CArray(y_val)

                    train_set = CDataset(cX_train, cy_train)
                    val_set = CDataset(cX_val, cy_val)

                    clf = CClassifierSVM(C=best_params['C'], kernel=CKernelRBF(gamma=best_params['gamma']))
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

                    n_poison = int(np.floor(X_train.shape[0] * p))
                    attack.n_points = n_poison

                    # Running attack
                    _, _, pois_examples, _ = attack.run(cX_val, cy_val)
                    X_pois = np.vstack([X_train, pois_examples.X.get_data()])
                    y_pois = np.concatenate([y_train, pois_examples.Y.get_data()])
                # Save poisoned data
                to_csv(X_pois, y_pois, cols, path_poison_data)

            clf_pois = CClassifierSVM(C=best_params['C'], kernel=CKernelRBF(gamma=best_params['gamma']))
            clf_pois.fit(X_pois, y_pois)

            pred_pois = clf_pois.predict(X_pois, y_pois)
            pred_test = clf_pois.predict(X_test, y_test)
            acc_train_pois = np.mean(pred_pois.get_data() == y_pois)
            acc_test_pois = np.mean(pred_test.get_data() == y_test)
        except Exception as e:
            print(e)
            acc_train_pois = 0
            acc_test_pois = 0
        time_elapse = time.time() - time_start
        print('Time: [{}] P-Rate {:.2f} Acc  P-train: {:.2f} C-test: {:.2f}'.format(
            time2str(time_elapse), p * 100, acc_train_pois * 100, acc_test_pois * 100))


        # Prepare score DataFrame
        path_poison_data_list.append(path_poison_data)
        accuracy_train_poison.append(acc_train_pois)
        accuracy_test_poison.append(acc_test_pois)

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
    df.to_csv(os.path.join(path_output, f'{dataname}_poison_svm_score.csv'), index=False)


if __name__ == '__main__':
    # Example:
    # python ./experiments/real/Step2_SecMLPoisoningSVM.py -f "data/real" -o "results/real" -d "breastcancer_std"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The path of the data')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('-o', '--output', type=str, default='results/real',
                        help='The output path for scores.')
    parser.add_argument('-s', '--step', type=float, default=0.1,
                        help='Spacing between values. Default=0.1')
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

    run_poison_attack(path_train, path_test, dataset, advx_range, filepath, output)
