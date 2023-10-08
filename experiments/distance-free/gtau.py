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

import numpy as np
from numpy import eye, ndarray, array, mean

from sortedness.local import sortedness


def M_to_direction_matrix(M):
    V = M.reshape((M.shape[0], 1, -1))
    H = M.reshape((1, -1, M.shape[1]))
    Sub = H - V
    Sign = np.sign(Sub)
    Sum = np.sum(Sign, axis=2)

    # Break majority vote ties by magnitude.
    mag = np.sum(Sub, axis=2)
    Sum[(Sum == 0) & (mag > 0)] = 1
    Sum[(Sum == 0) & (mag < 0)] = -1
    return np.sign(Sum)


def gtau(i, X, X_):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
        X_ = X_.to_numpy()
    n = len(X) - 1
    x = X[i] if isinstance(X, ndarray) else X.iloc[i]
    x_ = X_[i] if isinstance(X_, ndarray) else X_.iloc[i]
    D = M_to_direction_matrix(abs(X - x))
    D_ = M_to_direction_matrix(abs(X_ - x_))
    eqs = (D == D_)  # | (D == 0) | (D_ == 0)
    Agreem = np.triu(eqs, k=1)
    Agreem[i] = 0
    Agreem[:, i] = 0
    a = np.count_nonzero(Agreem)
    deno = n * (n - 1) / 2
    return (2 * a - deno) / deno


def global_gtau(X, X_):
    return array([gtau(i, X, X_) for i in range(len(X))])


n = 700
me = (5, 5, 5)
cov = eye(3)
rnd = np.random.default_rng(seed=6)

X = rnd.multivariate_normal(me, cov, size=n)
X_ = X.copy()
rnd.shuffle(X_)

X = np.sort(rnd.multivariate_normal(me, cov, size=n), axis=0)
# X_ = array(list(reversed(X)))
from sklearn.decomposition import PCA

X_ = PCA(n_components=3).fit_transform(X)

X = np.vstack([[0, 0, 0], X])
X_ = np.vstack([[0, 0, 0], X_])

A, B = X, X_
print(A)
# A, B = X[:, :1], X_[:, :1]
g = gtau(0, A, B)
print(g)
print(mean(global_gtau(A[1:], B[1:])))
print(mean(sortedness(A[1:], B[1:])))
# t=min([timeit(lambda :gtau(0, A, B), number=1) for _ in range(20)])
# print(t)
