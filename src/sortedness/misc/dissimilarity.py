#  Copyright (c) 2024. Davi Pereira dos Santos
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
from numpy import sqrt, zeros, mean
from numpy.linalg import norm

from sortedness.local import gaussian_np


def d_(X, a, a_b, diagl, w):
    D = abs(X - a)
    # so = np.sort(D, axis=0)
    # m = np.where(so > 0, so, so.max()).min(axis=0)
    # A = D / (a_b + m)
    A = D / a_b
    nm = norm(A, axis=1)
    if w is None:
        # return sum(diagl / (diagl + nm))  # grau de "vizinhância"
        return mean(diagl / (diagl + nm))  # distância normalizada pelo tamanho do dataset
    if w == "gaussian":
        return mean(gaussian_np(nm, sigma=diagl))
    raise Exception(f"Unknown 'w': '{w}'")


def d(X, a, b, diagl, w):
    a_b = abs(a - b)
    return (d_(X, a, a_b, diagl, w) + d_(X, b, a_b, diagl, w)) / 2


def matrix(X, w=None):
    """
    Dissimilarity matrix

    >>> from numpy.random import default_rng
    >>> from scipy.spatial.distance import cdist
    >>> rnd = default_rng(0)
    >>> X = rnd.random(size=(5, 2))
    >>> X
    array([[0.63696169, 0.26978671],
           [0.04097352, 0.01652764],
           [0.81327024, 0.91275558],
           [0.60663578, 0.72949656],
           [0.54362499, 0.93507242]])
    >>> M = matrix(X)
    >>> M
    array([[0.        , 0.50142118, 0.57556909, 0.41934944, 0.54175426],
           [0.50142118, 0.        , 0.69990897, 0.67337067, 0.67085368],
           [0.57556909, 0.69990897, 0.        , 0.51254106, 0.3438614 ],
           [0.41934944, 0.67337067, 0.51254106, 0.        , 0.44706731],
           [0.54175426, 0.67085368, 0.3438614 , 0.44706731, 0.        ]])
    >>> D = cdist(X, X)
    >>> D
    array([[0.        , 0.64756625, 0.66670358, 0.46070902, 0.67180117],
           [0.64756625, 0.        , 1.18307512, 0.91010904, 1.0470831 ],
           [0.66670358, 1.18307512, 0.        , 0.27619136, 0.27056718],
           [0.46070902, 0.91010904, 0.27619136, 0.        , 0.2150158 ],
           [0.67180117, 1.0470831 , 0.27056718, 0.2150158 , 0.        ]])
    >>> from scipy.stats import rankdata, kendalltau
    >>> rankdata(M, axis=1) - rankdata(D, axis=1)
    array([[ 0.,  0.,  1.,  0., -1.],
           [ 0.,  0.,  0.,  1., -1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [-2.,  0.,  1.,  0.,  1.],
           [ 0.,  0., -1.,  1.,  0.]])
    >>> [round(kendalltau(m, d)[0],3) for m, d in zip(M,D)]
    [0.8, 0.8, 1.0, 0.6, 0.8]

    >>> M = matrix(X, w="gaussian")
    >>> M
    array([[0.        , 0.41939533, 0.61125161, 0.35919401, 0.53977294],
           [0.41939533, 0.        , 0.8305749 , 0.77403604, 0.77507507],
           [0.61125161, 0.8305749 , 0.        , 0.4825503 , 0.32130614],
           [0.35919401, 0.77403604, 0.4825503 , 0.        , 0.35915179],
           [0.53977294, 0.77507507, 0.32130614, 0.35915179, 0.        ]])
    >>> from scipy.stats import rankdata, kendalltau
    >>> rankdata(M, axis=1) - rankdata(D, axis=1)
    array([[ 0.,  0.,  1.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  1.,  0.,  0.],
           [ 0.,  0., -1.,  1.,  0.]])
    >>> [round(kendalltau(m, d)[0],3) for m, d in zip(M,D)]
    [0.8, 1.0, 1.0, 0.8, 0.8]
    >>> rnd = default_rng(0)
    >>> X = rnd.random(size=(20, 2))
    >>> M = matrix(X)
    >>> D = cdist(X, X)
    >>> [round(kendalltau(m, d)[0],3) for m, d in zip(M,D)]
    [0.653, 0.568, 0.442, 0.632, 0.684, 0.411, 0.453, 0.621, 0.484, 0.558, 0.589, 0.684, 0.674, 0.537, 0.621, 0.695, 0.547, 0.453, 0.663, 0.526]
    """
    diagl = sqrt(len(X.shape))
    n = len(X)
    D = zeros((n, n))
    for i in range(n):
        a = X[i]
        for j in range(i + 1, n):
            b = X[j]
            D[i, j] = D[j, i] = d(X, a, b, diagl, w)
    return D
