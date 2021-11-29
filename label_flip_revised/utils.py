import datetime
import json
import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def time2str(time_elapsed):
    return time.strftime("%Hh%Mm%Ss", time.gmtime(time_elapsed))


def transform_label(y, target=-1):
    """Transform binary labels from {0, 1} to {-1, 1}. If target is 0, 
    transform them back to {0, 1}"""
    assert target in [-1, 0], f'Expect target to be -1 or 0, got {target}.'
    outputs = np.copy(y)
    neg_lbl = 0 if target == -1 else -1
    idx_neg_lbl = np.where(outputs == neg_lbl)[0]
    outputs[idx_neg_lbl] = target
    return outputs


def flip_binary_label(y, idx, use_neg_label=False):
    """Flip binary labels with given indices."""
    y_flip = np.copy(y)
    y_flip[y_flip == 0] = -1
    y_flip[idx] = - y_flip[idx]
    if use_neg_label:
        return y_flip
    y_flip[y_flip == -1] = 0
    return y_flip


def create_dir(path):
    """Create directory if the input path is not found."""
    if not os.path.exists(path):
        logger.info('Creating directory:', path)
        os.makedirs(path)


def open_csv(path_data, label_name='y'):
    """Read data from a CSV file, return X, y and column names."""
    logger.info('Load from:', path_data)
    df_data = pd.read_csv(path_data)
    y = df_data[label_name].to_numpy()
    df_data = df_data.drop([label_name], axis=1)
    cols = df_data.columns
    X = df_data.to_numpy()
    return X, y, cols


def to_csv(X, y, cols, path_data):
    """Save data into a CSV file."""
    logger.info('Save to:', path_data)
    df = pd.DataFrame(X, columns=cols)
    df['y'] = y
    df.to_csv(path_data, index=False)


def to_json(data_dict, path):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    with open(path, 'w') as file:
        logger.info('Save to:', path)
        json.dump(data_dict, file, default=converter)


def open_json(path):
    """Read JSON file."""
    try:
        with open(path, 'r') as file:
            data_json = json.load(file)
            return data_json
    except:
        logger.error(f'Cannot open {path}')
