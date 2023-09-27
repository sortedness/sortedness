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
from numpy import vstack, arange, array, argsort
from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.stats import weightedtau

from sortedness.local import remove_diagonal

seed, xmax, ymax, n, maxdims = 13, 1, 1, 300, 4  # critério de interrupção está errado pois exemplos x5 e x6 nunca aceitam todos os vizinhos.


def spiral(n):
    a = .11
    phi = arange(0, 100 * np.pi, 0.39)
    x = a * phi * np.cos(phi)
    y = a * phi * np.sin(phi)
    z = a * phi * np.cos(phi) * np.sin(phi)
    return array(list(zip(x, y, z)))[:n]


for dims in range(1, maxdims + 1):
    print(dims)
    rnd = np.random.default_rng(seed)
    xs = [rnd.uniform(0, xmax, n) for i in range(4)]
    X = vstack(xs).T

    X = spiral(n)

    rnd.shuffle(X)
    D2 = cdist(X, X, 'sqeuclidean')

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    ax.scatter(X[:1, 0], X[:1, 1], X[:1, 2], c=[1])

    for idx in range(5, 7):  # X.shape[0]):
        idxs = argsort(D2[idx])
        X = X[idxs]
        D2 = cdist(X, X, 'sqeuclidean')

        D = remove_diagonal(np.sqrt(D2))
        U2 = D2[0, 1]
        U = sqrt(U2)
        Z = np.zeros((X.shape[0], dims))
        Z[0] = np.array([0] * dims)
        Z[1] = np.array([U] + [0] * (dims - 1))


        def f(p_, ps, ds):
            lst = [norm(p_ - p) - d for p, d in zip(ps, ds)]
            return lst


        for i in range(2, X.shape[0]):
            r = least_squares(f, (0,) * dims, args=[Z[:i], D[i, :i]])
            # print(r)
            Z[i] = r.x
            # print(f"{r.cost}")
            Dz = remove_diagonal(cdist(Z[:i + 1], Z[:i + 1]))
            # worst = np.max(np.abs(D[:i+1, :i] - Dz))
            mn0 = np.mean(Dz[0])
            # print(worst,mn, r.cost)
            worst0 = max(abs(array(r.fun)))
            worst = weightedtau(-D[0, :i], -Dz[0])[0]
            # print((D[i, :i], Dz[0]))
            mn = 0.999999999
            # print(worst, mn, len(r.fun))
            if worst < mn and worst0 > mn0:
                break
        print(f"x_{idx} {dims}-d max neighborhood: {i} .   {worst0} {worst}")

    # print(np.min(D), np.max(D))
    # print(np.min(remove_diagonal(cdist(Z, Z))), np.max(remove_diagonal(cdist(Z, Z))))
    # print(np.max(cdist(X, X) - cdist(Z, Z)))
    print()
    # print(cdist(X[:4], X[:4]))
    # print()
    # print(cdist(Z[:4], Z[:4]))
    # print()
    # print()
