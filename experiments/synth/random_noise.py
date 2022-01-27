import argparse
import os
import time
import warnings
from pathlib import Path
from glob import glob 
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.preprocessing import StandardScaler

from label_flip_revised import flip_random
from label_flip_revised.utils import (create_dir, open_csv, open_json, to_csv,
                                      to_json, transform_label)


STEP = 0.05

def gen_random_labels(path_data, path_output, advx_range):
    files = sorted(glob(os.path.join(path_data, '*_clean_train.csv')))
    print(f'# of files: {len(files)}')
    for file in files:
        X, y, cols = open_csv(file, label_name='y')
        filename = Path(file).stem[:-len('_clean_train')]

        for p in tqdm(advx_range):
            y_flip = flip_random(y, p)
            path_output_data = os.path.join(path_output, f'{filename}_random_{p:.2f}.csv')
            to_csv(X, y_flip, cols, path_output_data)


# Example:
# python ./experiments/real/random_noise.py 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, default='./data/synth/train',
                        help='The file path of the data')
    parser.add_argument('-o', '--output', type=str, default='./data/synth/rand',
                        help='The output path')
    parser.add_argument('-s', '--step', type=float, default=STEP,
                        help='Spacing between values. Default=0.1')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval. Default=0.41')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    output = str(Path(args.output).absolute())
    step = args.step
    max_ = args.max

    advx_range = np.arange(0, max_, step)[1:]  # Remove 0%

    print(f'Path: {filepath}')
    print(f'Range: {advx_range}')

    # Create directory if not exist
    create_dir(os.path.join(output))


    gen_random_labels(path_data=filepath,
                      path_output=output,
                      advx_range=advx_range)
