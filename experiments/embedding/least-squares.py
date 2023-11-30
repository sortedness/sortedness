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
from matplotlib import pyplot as plt
from numpy import random, eye
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

n = 500
smoothness_ranking, lambd, decay = [5], [5], 0
batch_size = [20]
update = 2
seed = 0
fs = 16
rad = 120
alpha = 0.5
delta = 0

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target

print(datax.shape)
print(datay.shape)

alph = datay

rnd = random.default_rng(seed)
X = datax[:n]
X = X.astype(np.float32)
idxs = list(range(n))
D = pdist(X)

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax0, ax1 = axs
ax0.cla()
layout = rnd.multivariate_normal([0, 0], eye(2), n)
ax0.scatter(layout[:, 0], layout[:, 1], s=rad, c=alph[idxs], alpha=alpha)
for j in range(min(n, 50)):
    ax0.text(layout[j, 0] + delta, layout[j, 1] + delta, alph[j], size=fs)
c = [0]


def f(X_, D):
    D_ = pdist(X_.reshape(n, 2))
    return [d - d_ for d, d_ in zip(D, D_)]


r = least_squares(f, (0,) * 2 * n, args=[D])
X_ = r.x.reshape(n, 2)

ax1.cla()
ax1.scatter(X_[:, 0], X_[:, 1], s=rad, c=alph[idxs], alpha=alpha)
for j in range(min(n, 50)):
    ax1.text(X_[j, 0] + delta, X_[j, 1] + delta, alph[j], size=fs)
smoothness_ranking[0] *= 1 - decay
lambd[0] *= 1 - decay

mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
plt.show()
