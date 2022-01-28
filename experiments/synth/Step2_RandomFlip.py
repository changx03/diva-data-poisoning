import argparse
import os
from pathlib import Path
from glob import glob
from tqdm import tqdm

import numpy as np

from label_flip_revised import flip_random
from label_flip_revised.utils import create_dir, open_csv, to_csv


def gen_random_labels(path_data, path_output, advx_range):
    files = sorted(glob(os.path.join(path_data, 'train', '*_train.csv')))
    print(f'# of files: {len(files)}')
    for file in tqdm(files):
        X, y, cols = open_csv(file, label_name='y')
        filename = Path(file).stem[:-len('_train')]

        for p in advx_range:
            if p == 0:
                y_flip = y
            else:
                y_flip = flip_random(y, p)
            path_output_data = os.path.join(path_output, f'{filename}_rand_{p:.2f}.csv')
            to_csv(X, y_flip, cols, path_output_data)


if __name__ == '__main__':
    # Example:
    # python ./experiments/synth/Step2_RandomFlip.py -f "data/synth/"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data')
    parser.add_argument('-s', '--step', type=float, default=0.05,
                        help='Spacing between values. Default=0.0.5')
    parser.add_argument('-m', '--max', type=float, default=0.41,
                        help='End of interval. Default=0.41')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    step = args.step
    max_ = args.max

    advx_range = np.arange(0, max_, step)

    print(f'Path: {filepath}')
    print(f'Range: {advx_range}')

    # Create directory if not exist
    path_output = os.path.join(filepath, 'rand')
    create_dir(path_output)

    gen_random_labels(path_data=filepath,
                      path_output=path_output,
                      advx_range=advx_range)
