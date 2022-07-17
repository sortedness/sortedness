from functools import partial
from itertools import repeat
from math import nan, sqrt

from numpy import eye, argsort
from numpy.random import shuffle, permutation
from scipy.stats import spearmanr, weightedtau, kendalltau
from sklearn.decomposition import PCA

from robustress.rank import rank_by_distances, rdist_by_index_lw, rdist_by_index_iw, euclidean__n_vs_1


# noinspection PyTypeChecker
def kruskal(X_a, X_b, return_pvalues=False):
    """
    Kruskal's "Stress Formula 1"

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = kruskal(original, original)
    >>> min(s), max(s), s
    (0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = kruskal(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = kruskal(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.081106807792, 0.347563916162, [0.295668173586, 0.319595012703, 0.235774667847, 0.081106807792, 0.298113447155, 0.180984791932, 0.182406641753, 0.155316001865, 0.200126083035, 0.157911876379, 0.347563916162, 0.256262170166])
    >>> pvalues
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]


    Parameters
    ----------
    X_a
        matrix with an instance by row in a given space (often the original one)
    X_b
        matrix with an instance by row in another given space (often the projected one)
    return_pvalues
        Add dummy p-values to result (NaNs)

    Returns
    -------

    """
    result, pvalues = [], []
    for a, b in zip(X_a, X_b):
        d_a = euclidean__n_vs_1(X_a, a)
        d_b = euclidean__n_vs_1(X_b, b)
        kru = sqrt(sum((d_a - d_b) ** 2) / sum(d_a ** 2))
        result.append(round(kru, 12))

    if return_pvalues:
        return result, [nan for _ in result]
    return result
