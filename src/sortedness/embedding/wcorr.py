# Based on https://pypi.org/project/wcorr/

import numpy as np
from scipy.stats import rankdata


# from torchsort import soft_rank, soft_sort


def wcov(x, y, w, ms):
    return np.sum(w * (x - ms[0]) * (y - ms[1]))


def pearson(x, y, w):
    mx, my = (np.sum(i * w) / np.sum(w) for i in [x, y])
    return wcov(x, y, w, [mx, my]) / np.sqrt(wcov(x, x, w, [mx, mx]) * wcov(y, y, w, [my, my]))


def spearman(x, y, w):
    return pearson(wrank(x, w), wrank(y, w), w)

def wrank(x, w):
    """
    >>> wrank([1, 2, 2, 3], [1, 1, 1, 1])
    array([1. , 2.5, 2.5, 4. ])
    >>> wrank([1, 2, 3, 4], [1, 1/2, 1/3, 1/4])
    array([1.        , 1.5       , 1.83333333, 2.08333333])
    """
    (unique, arr_inv, counts) = np.unique(rankdata(x), return_counts=True, return_inverse=True)
    a = np.bincount(arr_inv, w)
    return (np.cumsum(a) - a)[arr_inv] + ((counts + 1) / 2 * (a / counts))[arr_inv]
