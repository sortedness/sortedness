from math import nan

from numpy import eye, where, setdiff1d
from numpy.random import shuffle
from sklearn.decomposition import PCA

from sortedness.rank import rank_by_distances


def continuity(X, X_, k=5, return_pvalues=False):
    """
    'continuity' of each point separately.

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = continuity(original, original)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = continuity(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = continuity(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.8, 1.0, [0.95, 0.8, 0.95, 1.0, 0.9, 0.95, 0.95, 1.0, 0.95, 1.0, 0.85, 0.9])
    >>> pvalues
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]


    Parameters
    ----------
    k
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    return_pvalues
        Add dummy p-values to result (NaNs)

    Returns
    -------
    List of values, one for each instance

    """
    return trustworthiness(X_, X, k, return_pvalues)


def trustworthiness(X, X_, k=5, return_pvalues=False):
    """
    'trustworthiness' of each point separately.

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = trustworthiness(original, original)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = trustworthiness(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = trustworthiness(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.75, 1.0, [0.8, 0.75, 0.9, 1.0, 0.85, 0.9, 0.95, 1.0, 0.95, 1.0, 0.85, 0.8])
    >>> pvalues
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]


    Parameters
    ----------
    k
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    return_pvalues
        Add dummy p-values to result (NaNs)

    Returns
    -------
    List of values, one for each instance

    """
    result, pvalues = [], []
    n = len(X)
    for a, b in zip(X, X_):
        ra = rank_by_distances(X, a, "min")
        rb = rank_by_distances(X_, b, "min")
        a_neighbors = where(ra <= k)
        b_neighbors = where(rb <= k)
        U = setdiff1d(b_neighbors, a_neighbors)
        r = 1 - 2 * sum(ra[U] - k) / k / (2 * n - 3 * k - 1)
        result.append(r)
    if return_pvalues:
        return result, [nan for _ in result]
    return result