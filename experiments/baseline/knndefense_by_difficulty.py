import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from label_flip_revised.knn_defense import KNNBasedDefense
from label_flip_revised.utils import create_dir, open_csv

ROOT = Path(os.getcwd())


def run_defense(metadata, path_output):
    df = metadata
    with open(path_output, 'a+') as file:
        file.write(','.join(['Path.Poison', 'Difficulty', 'Rate', 'Similarity']) + '\n')
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i]
            difficulty = row['Data'].split('_')[0]
            path_poison = row['Path.Poison']
            rate = row['Rate']

            # Load dataset
            X, y, _ = open_csv(path_poison)

            # Run defense
            defense = KNNBasedDefense(k=5, eta=0.5)
            X_sanitized, y_sanitized = defense.run(X, y)
            assert np.array_equal(X, X_sanitized)
            similarity = defense.eval(y, y_sanitized)

            row_output = ','.join([path_poison, difficulty, f'{rate:.2f}', f'{similarity:.4f}'])
            file.write(row_output + '\n')


if __name__ == '__main__':
    """Example:
    python ./experiments/baseline/knndefense_by_difficulty.py -i ./results/synth/synth_alfa_svm_score.csv -o ./results/synth/baseline/synth_alfa_svm_knndefense.csv
    python ./experiments/baseline/knndefense_by_difficulty.py -i ./results/synth/synth_falfa_nn_score.csv -o ./results/synth/baseline/synth_falfa_nn_knndefense.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The input CSV file that contains `Path.Train` and `Rate` columns.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The full name and directory of the output file')
    args = parser.parse_args()
    path_input = Path(args.input).absolute()
    path_output = Path(args.output).absolute()

    print('Root:', ROOT)

    # For testing only
    # path_input = Path(os.path.join(ROOT, 'results', 'synth', 'synth_alfa_svm_score.csv'))
    # path_output = Path(os.path.join(ROOT, 'results', 'synth', 'baseline', 'synth_alfa_svm_knndefense.csv'))

    create_dir(path_output.parent)

    print('Input:', path_input)
    print('Output:', path_output)
    if os.path.exists(path_output):
        print('Warning: Find existing output. Appending mode is on!')

    # Step1: Load CSV file to a DataFrame, get metadata for training sets;
    df = pd.read_csv(path_input)

    # Step2: Run KNN defense on each dataset, and save the results;
    run_defense(df, path_output)
