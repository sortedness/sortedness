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

from math import nan

import numpy as np
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
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = continuity(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = continuity(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.8, 1.0, array([0.95, 0.8 , 0.95, 1.  , 0.9 , 0.95, 0.95, 1.  , 0.95, 1.  , 0.85,
           0.9 ]))
    >>> pvalues
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])


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
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = trustworthiness(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = trustworthiness(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.75, 1.0, array([0.8 , 0.75, 0.9 , 1.  , 0.85, 0.9 , 0.95, 1.  , 0.95, 1.  , 0.85,
           0.8 ]))
    >>> pvalues
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])


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
    result = np.array(result)
    if return_pvalues:
        return result, np.array([nan for _ in result])
    return result
