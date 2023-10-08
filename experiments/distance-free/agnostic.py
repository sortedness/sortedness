#  Copyright (c) 2023. Davi Pereira dos Santos
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
from timeit import timeit

import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def snrank(X, i=None, j=None, mean=True):
    """
    Symmetric Neighborhood Rank

    Mean value between neighborhood rank and its reciprocal.
    Intended to be used as dissimilarity measure.


    Parameters
    ----------
    X
        Instance matrix
    i
        Index of an instance
    j
        Index of another instance

    Returns
    -------
    When `i` and `j` are provided, return the agnostic distance between them.
    Otherwise, return the Symmetric Neighborhood Rank for `X`.


    >>> import numpy as np
    >>> n = 5
    >>> me = (5, 5, 5)
    >>> cov = np.eye(3)
    >>> rnd = np.random.default_rng(seed=60)
    >>> X = rnd.multivariate_normal(me, cov, size=n)
    >>> np.round(snrank(X), 2)
    array([[0.  , 2.67, 6.67, 4.67, 4.33],
           [2.67, 0.  , 7.67, 6.67, 5.  ],
           [6.67, 7.67, 0.  , 3.  , 5.67],
           [4.67, 6.67, 3.  , 0.  , 3.67],
           [4.33, 5.  , 5.67, 3.67, 0.  ]])
    >>> np.round(snrank(X, 1, 3), 2)
    6.67
    """
    if i is None:
        V = X.reshape((X.shape[0], 1, -1))
        H = X.reshape((1, -1, X.shape[1]))
        Δ = np.abs(V - H)
        R = rankdata(Δ, method="dense", axis=1) - 1
        Rt = R.transpose(1, 0, 2)
        AD = R + Rt
        if not mean:
            return AD
        μ = np.mean(AD, axis=2)
        return μ
    else:
        if j is None:  # pragma: no cover
            raise Exception(f"Missing index `j`.")
        xi, xj = X[i], X[j]
        Ri = rankdata(np.abs(X - xi), method="dense", axis=0) - 1
        Rj = rankdata(np.abs(X - xj), method="dense", axis=0) - 1
        ad = Ri[j] + Rj[i]
        μ = np.mean(ad)
        return μ


def adist():
    """
    Agnostic distance

    This generates points in a metric space from a Symmetric Neighborhood Rank by using INDSCAL

    Returns
    -------

    """
    raise Exception(f"Fixing metric violation still under research. Here they handle it as noise,"
                    f"and minimize the number of fixed points: https://personal.utdallas.edu/~bar150630/mvd.pdf"
                    f"We need something that changes all points equally, to preserve the overall structure.")


def inc_diss_wrong(X: ndarray):
    """
    Intersected Neighbors Count (INC dissimilarity):  inc(a,b) = c + 1

    `c`: number of points inside hyperspheres intersection centered respectively at points `a` and `b`, both with radius `ab`

    >>> import numpy as np
    >>> n = 5
    >>> me = (5, 5, 5)
    >>> cov = np.eye(3)
    >>> rnd = np.random.default_rng(seed=60)
    >>> X = rnd.multivariate_normal(me, cov, size=n)
    >>> inc_diss(X)
    array([[0., 1., 3., 2., 1.],
           [1., 0., 4., 3., 2.],
           [3., 4., 0., 1., 2.],
           [2., 3., 1., 0., 1.],
           [1., 2., 2., 1., 0.]])
    """
    n = X.shape[0]
    D = cdist(X, X)
    for i in range(n):  # very slow
        for j in range(i + 1, n):
            r = D[i, j]

            # Slowest part.
            near_i = D[i] < r
            near_j = D[j] < r

            intersec = near_i & near_j
            c = np.count_nonzero(intersec) + 1
            D[i, j] = c
            D[j, i] = c
    return D


def inc_diss(X: ndarray):
    """
    Intersected Neighbors Count (INC dissimilarity):  inc(a,b) = c + 1

    `c`: number of points inside hyperspheres intersection centered respectively at points `a` and `b`, both with radius `ab`

    >>> import numpy as np
    >>> n = 5
    >>> me = (5, 5, 5)
    >>> cov = np.eye(3)
    >>> rnd = np.random.default_rng(seed=60)
    >>> X = rnd.multivariate_normal(me, cov, size=n)
    >>> inc_diss(X)
    array([[0., 1., 3., 2., 1.],
           [1., 0., 4., 3., 2.],
           [3., 4., 0., 1., 2.],
           [2., 3., 1., 0., 1.],
           [1., 2., 2., 1., 0.]])
    """
    # 288s for n=7000
    n = X.shape[0]
    D = cdist(X, X, metric="sqeuclidean")
    R = np.zeros(D.shape)
    for i in range(n):
        row_i = D[i:i + 1, :]
        rs = row_i.T[i + 1:]
        sphere_i = row_i < rs
        sphere_js = D[i + 1:] < rs
        intersec = sphere_i & sphere_js
        counts = np.count_nonzero(intersec, axis=1) + 1
        R[i:i + 1, i + 1:] = counts.reshape(1, -1)
        R[i + 1:, i:i + 1] = counts.reshape(-1, 1)
    return R


N = 7000
me = (5, 5, 5)
cov = np.eye(3)
rnd = np.random.default_rng(seed=60)
X = rnd.multivariate_normal(me, cov, size=N)
print(min([timeit(lambda: inc_diss(X), number=1) for _ in range(3)]))
# print(min([timeit(lambda: inc_diss_wrong(X), number=1) for _ in range(3)]))
exit()
