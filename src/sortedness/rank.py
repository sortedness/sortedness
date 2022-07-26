#  Copyright (c) 2022. Davi Pereira dos Santos
#  This file is part of the sortedness project.
#  Please respect the license - more about this in the section (*) below.
#
#  sortedness is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sortedness is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
#
#  (*) Removing authorship by any means, e.g. by distribution of derived
#  works or verbatim, obfuscated, compiled or rewritten versions of any
#  part of this work is illegal and it is unethical regarding the effort and
#  time spent here.
#

from typing import Union

from numpy import argsort
from numpy.linalg import norm
from scipy.stats import rankdata


def rdist_by_index_lw(indices, normalized=False, **kwargs):
    """
    Calculate distance between two rankings
    E.g.:
        d(abc, cba) = d(abc, cab)
        >>> rdist_by_index_lw([2, 1, 0]) == rdist_by_index_lw([1, 2, 0])
        True

        d(abc, acb) < d(abc, bac)
        >>> rdist_by_index_lw([0, 2, 1]) < rdist_by_index_lw([1, 0, 2])
        True

    For performance reasons, one of them (called `original`)
     is given as a transformation of the other (called `new`).

    >>> rdist_by_index_lw([2, 1, 5, 4, 3, 0], normalized=False)
    3.3
    >>> rdist_by_index_lw([1, 2, 5, 4, 3, 0], normalized=False)
    3.3
    >>> rdist_by_index_lw([1, 2, 5, 0, 4, 3], normalized=False)
    2.9
    >>> rdist_by_index_lw([0, 1, 2, 3, 4, 5])
    0
    >>> rdist_by_index_lw([5, 4, 3, 2, 1, 0], normalized=True)
    -1.0
    >>> round(rdist_by_index_lw([5, 4, 3, 2, 1, 0], normalized=False), 3)
    4.967
    >>> round(rdist_by_index_lw([0, 1, 2, 3, 5, 4], normalized=False), 3)
    0.333

    Parameters
    ----------
    indices
        Map between original position (index) and new position (value)
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
        This makes the measure dependent on dataset size
    Returns
    -------
        Distance between original and new rankings


    """
    costs = [1 / x for x in range(2, len(indices) + 1)]
    # TODO cache costs
    total = 0
    for i, idx in enumerate(indices):
        start, end = (i, idx) if i <= idx else (idx, i)
        cost = sum(costs[start:end])
        total += cost
    if total and normalized:
        total /= rdist_by_index_lw(list(range(len(indices) - 1, -1, -1)), normalized=False)
        total = 1 - 2 * total
    # TODO: calculate denominator analytically
    return round(total, 10)


def rdist_by_index_iw(indices, normalized=False, decay: Union[float, int] = 0):
    """
    Calculate distance between two rankings
    E.g.:
        d(abc, cba) = d(abc, cab)
        >>> rdist_by_index_iw([2, 1, 0]) == rdist_by_index_iw([1, 2, 0])
        True

        d(abc, acb) < d(abc, bac)
        >>> rdist_by_index_iw([0, 2, 1]) == rdist_by_index_iw([1, 0, 2])
        True

    For performance reasons, one of them (called `original`)
     is given as a transformation of the other (called `new`).

    >>> rdist_by_index_iw([2, 1, 5, 4, 3, 0])
    12
    >>> rdist_by_index_iw([1, 2, 5, 4, 3, 0])
    12
    >>> rdist_by_index_iw([1, 2, 5, 0, 4, 3])
    10
    >>> rdist_by_index_iw([0, 1, 2, 3, 4, 5])
    0
    >>> rdist_by_index_iw([5, 4, 3, 2, 1, 0], normalized=True)
    -1.0
    >>> rdist_by_index_iw([5, 4, 3, 2, 1, 0])
    18
    >>> rdist_by_index_iw([0, 1, 2, 3, 5, 4])
    2
    >>> rdist_by_index_iw([2, 1, 5, 4, 3, 0], decay=0.1)
    8.76755
    >>> rdist_by_index_iw([1, 2, 5, 4, 3, 0], decay=0.1)
    8.66755
    >>> rdist_by_index_iw([1, 2, 5, 0, 4, 3], decay=0.1)
    7.69798
    >>> rdist_by_index_iw([0, 1, 2, 3, 4, 5], decay=0.1)
    0.0
    >>> rdist_by_index_iw([5, 4, 3, 2, 1, 0], decay=0.1, normalized=True)
    -1.0
    >>> rdist_by_index_iw([5, 4, 3, 2, 1, 0], decay=0.1)
    14.15975
    >>> rdist_by_index_iw([0, 1, 2, 3, 5, 4], decay=0.1)
    1.24659
    >>> rdist_by_index_iw([0, 1, 2, 3, 5, 4]) == rdist_by_index_iw([1, 0, 2, 3, 4, 5])
    True

    Parameters
    ----------
    indices
        Map between original position (index) and new position (value)
    decay
        Decay factor to make the instance weight lesser than the previous instance in the ranking
        `decay=0` means uniform weights
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
        This makes the measure dependent on dataset size
    Returns
    -------
        Distance between original and new rankings


    """
    total = 0
    for i, idx in enumerate(indices):
        start, end = (i, idx) if i <= idx else (idx, i)
        w = (1 - decay) ** i
        cost = (end - start) * w
        total += cost
    if total and normalized:
        # TODO: calculate denominator analytically
        total /= rdist_by_index_iw(list(range(len(indices) - 1, -1, -1)), decay=decay, normalized=False)
        total = 1 - 2 * round(total, 10)
    return total


def rdist(a, b, normalized=False, f=rdist_by_index_iw):
    """
    >>> rdist([0, 1, 2, 3], [0, 1, 2, 3], f=rdist_by_index_lw)
    0
    >>> rdist([0, 1, 2, 3], [1, 0, 2, 3], f=rdist_by_index_lw)
    1.0
    >>> rdist([0, 1, 2, 3], [0, 1, 3, 2], f=rdist_by_index_lw)
    0.5
    >>> rdist([1, 0, 2, 3], [0, 1, 3, 2], f=rdist_by_index_lw)
    1.5
    >>> rdist([0, 1, 2, 3], [0, 1, 2, 3])
    0
    >>> rdist([0, 1, 2, 3], [1, 0, 2, 3])
    2
    >>> rdist([0, 1, 2, 3], [0, 1, 3, 2])
    2
    >>> rdist([1, 0, 2, 3], [0, 1, 3, 2])
    4

    Parameters
    ----------
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
    a
        List of numbers intended to be a rank
    b
        List of numbers intended to be a rank
    f
        Index-based distance function

    Returns
    -------

    """
    ranks_a = rankdata(a, method="ordinal") - 1
    ranks_b = rankdata(b, method="ordinal") - 1
    ranks = ranks_b[argsort(ranks_a)]
    return f(argsort(ranks), normalized=normalized)


def rank_by_distances(X, instance, method="average"):
    distances = euclidean__n_vs_1(X, instance)
    return rankdata(distances, method=method) - 1


def euclidean__n_vs_1(X, instance):
    return norm(X - instance, axis=1, keepdims=True)


def differences__n_vs_1(X, instance):
    """
    >>> import numpy as np
    >>> differences__n_vs_1(np.array([[1,0],[1,2],[2,1]]), [1,0])
    array([[0, 0],
           [0, 2],
           [1, 1]])


    Parameters
    ----------
    X
    instance

    Returns
    -------
    Absolute differences between attributes.
    """
    return abs(X - instance)


def neighbors(X, k=None):
    """
    >>> import numpy as np
    >>> neighbors(np.array([[1,2],[2,3],[5,4],[2,3],[2,1]]), 3)
    [array([[0],
           [1],
           [3]]), array([[1],
           [3],
           [0]]), array([[2],
           [1],
           [3]]), array([[1],
           [3],
           [0]]), array([[4],
           [0],
           [1]])]

    Parameters
    ----------
    X
        Matrix: one point/instance per row
    k
        Limit on the number of neighbors

    Returns
    -------
    M
        Matrix: one list of neighbors (instance index) per row

    """
    result = []
    for a in X:
        ds = euclidean__n_vs_1(X, a)
        idxs = argsort(ds, axis=0)
        r = idxs if k is None else idxs[:k]
        result.append(r)
    return result
