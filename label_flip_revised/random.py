"""Implement Random Flip Strategy 
"""
import math

import numpy as np

from .utils import flip_binary_label


def flip_random(y, p):
    """Randomly flip a percentage of binary labels"""
    n = y.shape[0]
    n_flip = int(math.floor(n * p))
    idx = np.random.permutation(n)[:n_flip]
    y_flip = flip_binary_label(y, idx, use_neg_label=False)
    return y_flip
