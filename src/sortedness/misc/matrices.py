import numpy as np
from numpy import argsort


def index(M, idxs):
    """
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> i = np.array([[0, 1, 2], [1, 2, 0], [1, 2, 0]])
    >>> index(a, i)
    array([[1, 2, 3],
           [5, 6, 4],
           [8, 9, 7]])
    """
    return M[np.arange(M.shape[0])[:, None], idxs]


def unindex(M, idxs):
    """
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> i = np.array([[0, 1, 2], [1, 2, 0], [1, 2, 0]])
    >>> ai = index(a, i)
    >>> unindex(ai, i)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    """
    return M[np.arange(M.shape[0])[:, None], argsort(idxs, kind="stable", axis=1)]
