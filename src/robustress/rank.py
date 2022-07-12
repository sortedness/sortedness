from numpy import argsort
from numpy.linalg import norm
from scipy.stats import rankdata


def dist(a, b, normalized=False):
    """
    >>> dist([0, 1, 2, 3], [0, 1, 2, 3])
    0
    >>> dist([0, 1, 2, 3], [1, 0, 2, 3])
    1.0
    >>> dist([0, 1, 2, 3], [0, 1, 3, 2])
    0.5
    >>> dist([1, 0, 2, 3], [0, 1, 3, 2])
    1.5

    Parameters
    ----------
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
    a
        List of numbers intended to be a rank
    b
        List of numbers intended to be a rank

    Returns
    -------

    """
    ranks_a = rankdata(a, method="ordinal") - 1
    ranks_b = rankdata(b, method="ordinal") - 1
    ranks = ranks_b[argsort(ranks_a)]
    return rank_dist__by_index(argsort(ranks), normalized=normalized)


def rank_dist__by_index(indices, normalized=False):
    """
    Calculate distance between two rankings
    E.g.:
        d(abc, cba) = d(abc, cab)
        >>> rank_dist__by_index([2, 1, 0]) == rank_dist__by_index([1, 2, 0])
        True

        d(abc, acb) < d(abc, bac)
        >>> rank_dist__by_index([0, 2, 1]) < rank_dist__by_index([1, 0, 2])
        True

    For performance reasons, one of them (called `original`)
     is given as a transformation of the other (called `new`).

    >>> rank_dist__by_index([2, 1, 5, 4, 3, 0], normalized=False)
    3.3
    >>> rank_dist__by_index([1, 2, 5, 4, 3, 0], normalized=False)
    3.3
    >>> rank_dist__by_index([1, 2, 5, 0, 4, 3], normalized=False)
    2.9
    >>> rank_dist__by_index([0, 1, 2, 3, 4, 5])
    0
    >>> rank_dist__by_index([5, 4, 3, 2, 1, 0], normalized=True)
    1.0
    >>> round(rank_dist__by_index([5, 4, 3, 2, 1, 0], normalized=False), 3)
    4.967
    >>> round(rank_dist__by_index([0, 1, 2, 3, 5, 4], normalized=False), 3)
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
        Distance between original


    """
    costs = [1 / x for x in range(2, len(indices) + 1)]
    total = 0
    for i, idx in enumerate(indices):
        start, end = (i, idx) if i <= idx else (idx, i)
        cost = sum(costs[start:end])
        total += cost
    if total and normalized:
        total /= rank_dist__by_index(list(range(len(indices) - 1, -1, -1)), normalized=False)
    # TODO: calculate denominator analytically
    return round(total, 10)


def rank_by_distances(m, instance):
    distances = norm(m - instance, axis=1, keepdims=True)
    return rankdata(distances, method="ordinal") - 1
