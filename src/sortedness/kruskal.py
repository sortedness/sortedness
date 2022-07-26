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

from functools import partial
from math import nan, sqrt

import numpy as np
from numpy import eye
from numpy.random import shuffle
from sklearn.decomposition import PCA

from sortedness.rank import rank_by_distances, euclidean__n_vs_1


# TODO: Stress majorization
# noinspection PyTypeChecker
def kruskal(X, X_, f=euclidean__n_vs_1, return_pvalues=False):
    """
    Kruskal's "Stress Formula 1"
    default: Euclidean

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = kruskal(original, original)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = kruskal(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = kruskal(original, projected, return_pvalues=True)
    >>> min(s), max(s), s
    (0.081106807792, 0.347563916162, array([0.29566817, 0.31959501, 0.23577467, 0.08110681, 0.29811345,
           0.18098479, 0.18240664, 0.155316  , 0.20012608, 0.15791188,
           0.34756392, 0.25626217]))
    >>> pvalues
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    >>> kruskal(original, projected)
    array([0.29566817, 0.31959501, 0.23577467, 0.08110681, 0.29811345,
           0.18098479, 0.18240664, 0.155316  , 0.20012608, 0.15791188,
           0.34756392, 0.25626217])
    >>> kruskal(original, projected, f=partial(rank_by_distances))
    array([0.35004235, 0.38755339, 0.17782169, 0.06286946, 0.27404163,
           0.1539981 , 0.23523598, 0.1088931 , 0.14058039, 0.1088931 ,
           0.33856241, 0.25147785])



    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    return_pvalues
        Add dummy p-values to result (NaNs)

    Returns
    -------

    """
    result, pvalues = [], []
    for a, b in zip(X, X_):
        d_a = f(X, a)
        d_b = f(X_, b)
        kru = sqrt(sum((d_a - d_b) ** 2) / sum(d_a**2))
        result.append(round(kru, 12))
    result = np.array(result)
    if return_pvalues:
        return result, np.array([nan for _ in result])
    return result
