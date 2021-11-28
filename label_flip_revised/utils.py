import time

import numpy as np


def time2str(time_elapsed):
    return time.strftime("%Hh%Mm%Ss", time.gmtime(time_elapsed))


def flip_binary_label(y, idx, use_neg_label=False):
    y_flip = np.copy(y)
    y_flip[y_flip == 0] = -1
    y_flip[idx] = - y_flip[idx]
    if use_neg_label:
        return y_flip
    y_flip[y_flip == -1] = 0
    return y_flip
