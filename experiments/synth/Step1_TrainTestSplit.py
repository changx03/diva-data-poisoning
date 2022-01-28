"""
Split training and test sets
"""
import argparse
import os
from glob import glob
from pathlib import Path

from sklearn.model_selection import train_test_split

from label_flip_revised.utils import create_dir, open_csv, to_csv

TEST_SIZE = 0.2


def split_data(path_data, path_output, test_size):
    path_list = glob(os.path.join(path_data, '*.csv'))
    print(f'Found {len(path_list)} datasets.')

    for p in path_list:
        X, y, cols = open_csv(p)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        dataname = Path(p).stem
        to_csv(X_train, y_train, cols, os.path.join(path_output, 'train', f'{dataname}_train.csv'))
        to_csv(X_test, y_test, cols, os.path.join(path_output, 'test', f'{dataname}_test.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', type=str, required=True,
                        help='The file path of the data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The output path')
    parser.add_argument('-t', '--testsize', type=float, default=TEST_SIZE,
                        help='The size of test set. Default=0.2')
    args = parser.parse_args()
    filepath = str(Path(args.filepath).absolute())
    output = str(Path(args.output).absolute())
    testsize = args.testsize

    print(f'Path: {filepath}')

    # Create directory if not exist
    create_dir(os.path.join(output, 'train'))
    create_dir(os.path.join(output, 'test'))

    split_data(filepath, output, testsize)
