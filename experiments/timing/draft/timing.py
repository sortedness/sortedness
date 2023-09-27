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

from sys import argv
from timeit import timeit

import numpy as np
from numpy import array, mean
from scipy.stats import rankdata
from sklearn.datasets import fetch_california_housing as dataset
from sklearn.decomposition import PCA

from sortedness.probabilistic import locglo, locgloth

print(argv)
n = int(argv[1])
m = int(argv[2])
S = int(argv[3])

X = dataset(return_X_y=True)[0][:min(n, 20640), :m]

print(X.shape)
X = np.repeat(X, n // len(X), axis=0)
n = len(X)
me = (0,) * m
cov = np.eye(m)
rnd = np.random.default_rng(seed=60)
X = X + rnd.multivariate_normal(me, cov, size=n)

print(X.shape)
print("------------")
X_ = PCA(n_components=4).fit_transform(X)
# X_ = TSNE(n_components=2, n_jobs=-1).fit_transform(X_)[:, :1]
print("ttttttttttttt")
# print(mean(stress(X, X_)))
# ps(X, X_)
# print(mean(sortedness(X, X_)))
# print()
# print(min([timeit(lambda: print(mean(stress(X, X_))), number=1) for _ in range(3)]))
print()
n = X.shape[0] - 1
p = array([1 / r for r in range(1, n + 1)])
s = sum(p)
p = p / s
p1 = array([1 / n for r in range(1, n + 1)])


def wrho(x, y):
    x = rankdata(x)
    y = rankdata(y)
    nom = 6 * sum(p * (x - y) ** 2)
    den = n * (n ** 2 - 1)
    return 1 - nom / den


# print(min([timeit(lambda: print(mean(sortedness(X, X_, f=kendalltau))), number=1) for _ in range(1)]))
# print(min([timeit(lambda: print(mean(sortedness(X, X_, f=wrho))), number=1) for _ in range(1)]))
# print(min([timeit(lambda: print(mean(sortedness(X, X_, symmetric=False))), number=1) for _ in range(1)]))
print()
# print("t prob", min([timeit(lambda: print(mean(ps(X, X_, weigher=cauchy))), number=1) for _ in range(1)]))
# print("t prob", min([timeit(lambda: print(mean(ps(X, X_))), number=1) for _ in range(1)]))
# print()
# print("t locgloq", min([timeit(lambda: print(mean(locgloq(X, X_, S=S))), number=1) for _ in range(1)]))
print()
print("t locgloth", min([timeit(lambda: print(mean(locgloth(X, X_))), number=1) for _ in range(1)]))
print()
print("t locglo", min([timeit(lambda: print(mean(locglo(X, X_))), number=1) for _ in range(1)]))
print()
# print(min([timeit(lambda: print(mean(sortedness(X, X_, symmetric=False, f=lambda x, y: WeightedCorr(x, y, w=p)("spearman")))), number=1) for _ in range(1)]))
# print(min([timeit(lambda: print(mean(ps(X, X_, f=spearmanr))), number=1) for _ in range(1)]))
# print("t sort", min([timeit(lambda: print(mean(sortedness(X, X_, weigher=cauchy))), number=1) for _ in range(1)]))
# print("t sort", min([timeit(lambda: print(mean(sortedness(X, X_))), number=1) for _ in range(1)]))
