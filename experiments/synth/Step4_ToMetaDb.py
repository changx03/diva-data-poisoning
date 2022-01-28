import argparse
import os
from pathlib import Path
from glob import glob

import pandas as pd

COL_NAMES = [
    'F1',
    'F1 SD',
    'F1v',
    'F2',
    'F3',
    'F4',
    'N1',
    'N2',
    'N2 SD',
    'N3',
    'N3 SD',
    'N4',
    'N4 SD',
    'T1',
    'T1 SD',
    'LSC',
    'L1',
    'L2',
    'L3',
    'T2',
    'T3',
    'T4',
    'C1',
    'C2',
    'Density',
    'ClsCoef',
    'Hubs',
    'Hubs SD',
    'Path.Train',
    'Path.Poison',
    'Path.Test',
    'Rate',
    'Train.Clean',
    'Test.Clean',
    'Train.Poison',
    'Test.Poison',
    'Data.Base',
]


def create_database(path_cm, path_score, path_output):
    cMeasure_list = glob(os.path.join(path_cm, '*.csv'))
    print(f'Found {len(cMeasure_list)} C-Measures files')

    # Load C-Measures
    df_cm = pd.DataFrame()
    for f in cMeasure_list:
        df_ = pd.read_csv(f)
        df_cm = pd.concat([df_cm, df_])
    # Set 'Data' name as indices
    df_cm = df_cm.set_index('Data')

    # Load Scores
    df_score = pd.read_csv(path_score)
    assert df_cm.shape[0] == df_score.shape[0], f'{df_cm.shape[0]} != {df_score.shape[0]}'

    df_score['Data.Base'] = df_score['Data']
    df_score['Data'] = df_score['Path.Poison'].apply(lambda x: x.split('/')[-1])
    df_score = df_score.set_index('Data')

    # Merge two DataFrames
    df_cm = pd.concat([df_cm, df_score], axis=1)

    # Drop NA
    print('# of columns before removing NA:', len(df_cm.columns))
    cols_not_na = df_cm.columns[df_cm.notna().any()].tolist()
    df_cm = df_cm[cols_not_na]
    print('# of columns after removing NA:', len(df_cm.columns))

    # Rename columns
    new_names_map = {df_cm.columns[i]: COL_NAMES[i] for i in range(len(COL_NAMES))}
    df_cm = df_cm.rename(new_names_map, axis=1)

    # Save result; Also save the index column
    print(f'Save database to: {path_output}')
    df_cm.to_csv(path_output, index=True)


if __name__ == '__main__':
    # Example:
    # python ./experiments/synth/Step4_ToMetaDb.py -c "results/synth/falfa_nn" -s "results/synth/synth_falfa_nn_score.csv" -o "results/synth/synth_falfa_nn_db.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cm', type=str, required=True,
                        help='The folder for C-Measures')
    parser.add_argument('-s', '--score', type=str, required=True,
                        help='The file for scores')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The full name and directory of the output file')
    args = parser.parse_args()
    path_cm = str(Path(args.cm).absolute())
    path_score = str(Path(args.score).absolute())
    path_output = str(Path(args.output).absolute())
    print(f'C-Measures Path: {path_cm}')
    print(f'Score Path: {path_score}')

    create_database(path_cm, path_score, path_output)
