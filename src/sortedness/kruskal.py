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
from itertools import repeat
from math import nan, sqrt

import numpy as np
from numpy import eye, argsort
from numpy.random import shuffle, permutation
from scipy.stats import spearmanr, weightedtau, kendalltau
from sklearn.decomposition import PCA

from sortedness.rank import rank_by_distances, rdist_by_index_lw, rdist_by_index_iw, euclidean__n_vs_1


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
    >>> kruskal(original, projected)
    [0.295668173586, 0.319595012703, 0.235774667847, 0.081106807792, 0.298113447155, 0.180984791932, 0.182406641753, 0.155316001865, 0.200126083035, 0.157911876379, 0.347563916162, 0.256262170166]
    >>> kruskal(original, projected, f=partial(rank_by_distances))
    [0.350042346394, 0.387553387882, 0.17782168979, 0.062869461346, 0.274041628643, 0.153998100702, 0.235235984448, 0.108893101296, 0.140580389279, 0.108893101296, 0.338562410685, 0.251477845385]



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
        kru = sqrt(sum((d_a - d_b) ** 2) / sum(d_a ** 2))
        result.append(round(kru, 12))
    result = np.array(result)
    if return_pvalues:
        return result, np.array([nan for _ in result])
    return result
