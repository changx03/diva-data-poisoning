"""Run kNN-based defense on synthetic datasets grouped by noise rate;
"""
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
    # df = metadata.iloc[:50].copy()  # Testing to top 50 datasets

    # Get noise rate
    noise_rate = df['Data'].apply(lambda x: float(x.split('_')[6][2:])).to_numpy()
    df['Noise.Rate'] = noise_rate

    with open(path_output, 'a+') as file:
        file.write(','.join(['Path.Poison', 'Noise', 'Rate', 'Similarity']) + '\n')
        for i in tqdm(range(df.shape[0])):
            row = df.iloc[i]
            noise = row['Noise.Rate']
            path_poison = row['Path.Poison']
            rate = row['Rate']

            # Load dataset
            X, y, _ = open_csv(path_poison)

            # Run defense
            defense = KNNBasedDefense(k=5, eta=0.5)
            X_sanitized, y_sanitized = defense.run(X, y)
            assert np.array_equal(X, X_sanitized)
            similarity = defense.eval(y, y_sanitized)

            row_output = ','.join([path_poison, f'{noise:.2f}', f'{rate:.2f}', f'{similarity:.4f}'])
            file.write(row_output + '\n')


if __name__ == '__main__':
    """Example:
    python ./experiments/baseline/knndefense_by_noise.py -i "./results/synth_noisy/synth_alfa_svm_score.csv" -o "./results/synth_noisy/baseline/synth_alfa_svm_knndefense.csv"
    python ./experiments/baseline/knndefense_by_noise.py -i "./results/synth_noisy/synth_falfa_nn_score.csv" -o "./results/synth_noisy/baseline/synth_falfa_nn_knndefense.csv"
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
    # path_input = Path(os.path.join(ROOT, 'results', 'synth_noisy', 'synth_alfa_svm_score.csv'))
    # path_output = Path(os.path.join(ROOT, 'results', 'synth_noisy', 'baseline', 'synth_alfa_svm_knndefense.csv'))

    create_dir(path_output.parent)

    print('Input:', path_input)
    print('Output:', path_output)
    if os.path.exists(path_output):
        print('Warning: Find existing output. Appending mode is on!')

    # Step1: Load CSV file to a DataFrame, get metadata for training sets;
    df = pd.read_csv(path_input)

    # Step2: Run KNN defense on each dataset, and save the results;
    run_defense(df, path_output)
