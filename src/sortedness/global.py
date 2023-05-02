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

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, ndarray
from numpy.random import permutation
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau


def global_pwsortedness(X, X_, parallel=True, parallel_n_trigger=10000, **parallel_kwargs):
    """
    Global pairwise sortedness (Λ𝜏1)

    # TODO?: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        This probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        Values can be lower due to the presence of ties, but only when the projection isn't prefect for all points.
        In the end, it might be even desired to penalize ties, as they don't exactly contribute to a stronger ordering and are (probabilistically) easier to be kept than a specific order.

    Parameters
    ----------
    X
        Original dataset or precalculated pairwise squared distances from pdist(X, metric="sqeuclidean")
    X_
        Projected points or precalculated pairwise squared distances from pdist(X, metric="sqeuclidean")
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    parallel_kwargs
        Any extra argument to be provided to pathos parallelization
    parallel_n_trigger
        Threshold to disable parallelization for small n values

    Returns
    -------
    (Λ𝜏1, p-value)
        The p-value considers the absence of order (Λ𝜏1 = 0) as the null hypothesis.

    >>> ll = [[i] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> global_pwsortedness(a, b)
    SignificanceResult(statistic=0.76, pvalue=1.6669837696943839e-34)
    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> global_pwsortedness(a, b)
    SignificanceResult(statistic=0.04141176470588235, pvalue=0.5044358739518093)
    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> from sklearn.decomposition import PCA
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)
    >>> global_pwsortedness(original, original, f=kendalltau, return_pvalues=True)
    SignificanceResult(statistic=1.0, pvalue=3.6741408919675163e-93)
    >>> global_pwsortedness(original, projected2, f=kendalltau)
    SignificanceResult(statistic=1.0, pvalue=3.6741408919675163e-93)
    >>> global_pwsortedness(original, projected1, f=kendalltau)
    SignificanceResult(statistic=0.7715617715617715, pvalue=5.240847664048334e-20)
    >>> global_pwsortedness(original, projectedrnd, f=kendalltau)
    SignificanceResult(statistic=-0.06107226107226107, pvalue=0.46847188611226276)
    """
    # TODO: parallelize pdist into a for?
    thread = lambda M: pdist(M, metric="sqeuclidean")
    npoints = len(X)
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    dists_X, dists_X_ = tmap(thread, [X, X_])
    return kendalltau(dists_X, dists_X_)


def cov2dissimilarity(M: ndarray):
    """

    Parameters
    ----------
    M

    Returns
    -------

    >>> import numpy as np
    >>> cov2dissimilarity(np.array([[1,2],[3,4]]))
    array([[ 0.,  1.],
           [-1.,  0.]])
    """
    variances = M.diagonal()
    dissimilarities = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dissimilarities[i, j] = variances[i] + variances[j] - 2 * M[i, j]
    return dissimilarities
