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
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import arange, array
from numpy.random import default_rng
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target
alphabet = datay
n = 100
idxs = list(range(n))
print(datax.shape, datay.shape)
seed = 0
torch.manual_seed(seed)
rnd = default_rng(seed)
X = datax[:n]
y = datay[:n]

X = X[:, ~(X == X[0, :]).all(0)]
mn = X.min(axis=0)
mx = X.max(axis=0)
X = (X - mn) / (mx - mn)


def spiral(n):
    phi = arange(0, 1000 * np.pi, 0.17)[:n]
    x = phi * np.cos(phi)
    y = phi * np.sin(phi)
    r = phi[-1] / 2
    z = 2 * np.sqrt(r ** 2 - (phi - r) ** 2)
    return array(list(zip(x, y, z)))


size = 150
color = y

size = list(map(lambda x: 10 + x ** .9, (range(n))))
color = range(n)
X = spiral(n)
epochs = 100
fs = {
    "PCA": lambda d: PCA(d).fit_transform(X),
    "MDS": lambda d: MDS(d, metric=True).fit_transform(X),
    "nMDS": lambda d: MDS(d, metric=False).fit_transform(X),
    "t-SNE": lambda d: TSNE(d).fit_transform(X),
    # "B00": lambda d: balanced_embedding(X, d, alpha=0, beta=0, epochs=epochs, hyperoptimizer=2), #bad
    # "B00_": lambda d: balanced_embedding_(X, d, alpha=0, beta=0, epochs=epochs, hyperoptimizer=2),
    # "B01": lambda d: balanced_embedding(X, d, alpha=0, beta=1, epochs=epochs, hyperoptimizer=2),
    # "B01_": lambda d: balanced_embedding_(X, d, alpha=0, beta=1, epochs=epochs, hyperoptimizer=2),
    # "B10": lambda d: balanced_embedding(X, d, alpha=1, beta=0, epochs=epochs, hyperoptimizer=2),#bad
    # "B10_": lambda d: balanced_embedding_(X, d, alpha=1, beta=0, epochs=epochs, hyperoptimizer=2),#bad
    # "B11": lambda d: balanced_embedding(X, d, alpha=1, beta=1, epochs=epochs, hyperoptimizer=2),
    # "B11_": lambda d: balanced_embedding_(X, d, alpha=1, beta=1, epochs=epochs, hyperoptimizer=2),
    # "B05": lambda d: balanced_embedding(X, d, alpha=0, beta=.5, epochs=epochs, hyperoptimizer=2),
    # "B05_": lambda d: balanced_embedding_(X, d, alpha=0, beta=.5, epochs=epochs, hyperoptimizer=2),
    # "B50": lambda d: balanced_embedding(X, d, alpha=.5, beta=0, epochs=epochs, hyperoptimizer=2),  # bad
    # "B50_": lambda d: balanced_embedding_(X, d, alpha=.5, beta=0, epochs=epochs, hyperoptimizer=2),
    # "B55": lambda d: balanced_embedding(X, d, alpha=.5, beta=.5, epochs=epochs, hyperoptimizer=2),
    # "B55_": lambda d: balanced_embedding_(X, d, alpha=.5, beta=.5, epochs=epochs, hyperoptimizer=2),
}
N = len(fs)

rows, cols = 12, 8
fig = plt.figure()
gs = GridSpec(rows, cols, figure=fig)
ax = fig.add_subplot(gs[:6, :4], projection="3d", title=f"Original")
ax.scatter(X[:, 0], X[:, 1], zs=0.001, zdir='z', c="lightgray", s=np.array(size) * .5, alpha=1)
ax.scatter(X[:, 0], X[:, 1], X[:, 2] + 0.002, c=color, s=size, alpha=.9, edgecolor="white")
ax.set_zlim(0, max(X[:, 2]))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.view_init(elev=45., azim=-45, roll=0)

j = 0
for i, (name, f) in zip([0, 3, 6, 6, 9, 9, 12, 12], fs.items()):
    c = 0 if j in [2, 4, 6, 8, 10, 12] else 4
    ax = fig.add_subplot(gs[i:2 + i, c:c + 2], projection="3d")
    ax.set_title(f"{name}", x=1, y=0.85)
    X_ = f(3)
    # ax.scatter(X_[:, 0], X_[:, 1], zs=0.001, zdir='z', c="lightgray", s=np.array(size) * .5, alpha=1)
    ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2] + 0.002, c=color, s=size, alpha=.9, edgecolor="white")
    ax.set_zlim(min(X_[:, 2]), max(X_[:, 2]))
    ax.view_init(elev=45., azim=-45, roll=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    for d in [2, 1]:
        g1 = gs[i:2 + i, c + 2:c + 4]
        g2 = gs[2 + i:3 + i, c:c + 4]
        ax = fig.add_subplot(g1 if d == 2 else g2)  # , title=f"{name} {d}D")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        X_ = f(d)
        y = X_[:, 1] if d == 2 else n * [0]
        ax.scatter(X_[:, 0], y, c=color, s=size, alpha=.9, edgecolor="white")
    j += 1

plt.show()
