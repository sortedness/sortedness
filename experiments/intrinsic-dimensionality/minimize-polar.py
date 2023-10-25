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
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from numpy import vstack, arange, array, argsort, ndarray
from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from sortedness.local import remove_diagonal

seed, xmax, ymax = 13, 1, 1
n, maxdims, spira = 100, 8, not True
points = 3
if spira:
    maxdims = 3


def spiral(n):
    a = .11
    phi = arange(0, 100 * np.pi, 0.39)
    x = a * phi * np.cos(phi)
    y = a * phi * np.sin(phi)
    z = a * phi * np.cos(phi) * np.sin(phi)
    return array(list(zip(x, y, z)))[:n]


for dims in range(1, maxdims + 1):
    print(f"{dims}-d")
    rnd = np.random.default_rng(seed)
    xs = [rnd.uniform(0, xmax, n) for i in range(maxdims + 1)]
    X = vstack(xs).T

    if spira:
        X = spiral(n)

    X = StandardScaler().fit_transform(X)

    rnd.shuffle(X)
    D2 = cdist(X, X, 'sqeuclidean')

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    # ax.scatter(X[:1, 0], X[:1, 1], X[:1, 2], c=[1])

    for idx in range(0, X.shape[0], n // points):
        idxs = argsort(D2[idx])
        X = X[idxs]
        D2 = cdist(X, X, 'sqeuclidean')

        D = remove_diagonal(np.sqrt(D2))
        U2 = D2[0, 1]
        U = sqrt(U2)
        Z = np.zeros((X.shape[0], dims))
        Z[0] = np.array([0] * dims)
        Z[1] = np.array([U] + [0] * (dims - 1))


        def f(alpha, rad, ps, ds):
            p_ = array([rad * np.cos(alpha), rad * np.sin(alpha)])
            lst = [norm(p_ - p) - d for p, d in zip(ps, ds)]
            return lst
        def f(p_, ps, ds):
            lst = [norm(p_ - p) - d for p, d in zip(ps, ds)]
            return lst


        olderr = None
        for i in range(2, X.shape[0]):
            r = least_squares(f, (0,) * dims, args=[Z[:i], D[i, :i]])
            DD: ndarray = D[i, :i]

            # mn = np.unique(np.sort(DD))
            # mn = mn[int(len(mn) * pct)]

            th = np.min(DD)
            err = np.max(abs(array(r.fun)))

            if err >= th:
                break
            Z[i] = r.x
            olderr = err

        if i == X.shape[0] - 1 or olderr is None:
            if olderr is None:
                i -= 1
            print(f"{i} neighbors can be represented in {dims}-d with error {err:.3f} < {th:.3f} for point x_{idx}")
        else:
            print(f"{i - 1} neighbors can be represented in {dims}-d with error {olderr:.3f} < {th:.3f} for point x_{idx}")

        if dims == 3:
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2])
            ax.scatter(Z[:1, 0], Z[:1, 1], Z[:1, 2], c=[1])
            plt.show()

    print()
