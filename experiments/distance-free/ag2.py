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

import torch
from matplotlib import pyplot as plt
from numpy.random import default_rng
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sortedness.embedding.sortedness_ import balanced_embedding_

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target
alphabet = datay
n = 300
idxs = list(range(n))
print(datax.shape, datay.shape)
seed = 0
torch.manual_seed(seed)
rnd = default_rng(seed)
X = datax[:n]

X = X[:, ~(X == X[0, :]).all(0)]
mn = X.min(axis=0)
mx = X.max(axis=0)
X = (X - mn) / (mx - mn)

size = list(map(lambda x: 2 + x * 3.0, (range(n))))
# X_ = MDS(metric=True).fit_transform(X)
# X_ = MDS(metric=False).fit_transform(X)
# X_ = balanced_embedding(X, epochs=120, hyperoptimizer=2)
X_ = balanced_embedding_(X, epochs=5, hyperoptimizer=2, beta=0)
# X_ = PCA().fit_transform(X)
# X_ = TSNE().fit_transform(X)

ax = plt.subplot(1, 1, 1)
ax.scatter(X_[:, 0], X_[:, 1], c=alphabet[idxs], s=300, alpha=.5, edgecolor="white")
for j in range(min(n, 50)):
    ax.text(X_[j, 0], X_[j, 1], alphabet[j], size=10)

plt.show()
