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

import numpy as np
import torch
from hdict import hdict, apply, cache
from matplotlib import pyplot as plt
from numpy import array
from numpy.random import permutation
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau, weightedtau
from shelchemy import sopen
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sortedness.attempts.wcorr import spearman
from sortedness.attempts.wcorrtorch import spearmangpt, spearman as softrho
from torch import tensor

n = 1_007


def get(n):
    print("fetching...")
    X = fetch_covtype(return_X_y=True)[0]
    print("ok")
    rnd = np.random.default_rng(seed=32141)
    rnd.shuffle(X)
    X = X[:n]
    Xp = PCA(n_components=2).fit_transform(X)
    Xt = X.copy()  # TSNE(n_components=2).fit_transform(X)
    rnd.shuffle(Xt)
    d = (d := pdist(X, metric="euclidean"))
    dp = (dp := pdist(Xp, metric="euclidean"))
    dt = (dt := pdist(Xt, metric="euclidean"))
    ddd = squareform(d)
    dddp = squareform(dp)
    dddt = squareform(dt)
    # dct = {"pca ρ": [], "tsne ρ": [], "pca $\\tau$": [], "tsne $\\tau$": [], "pca wρ": [], "tsne wρ": [], "pca w$\\tau$": [], "tsne w$\\tau$": [], "pca wsoftρ": [], "tsne wsoftρ": []}
    # dct = {"pca w$\\tau$": [], "tsne w$\\tau$": [], "pca wρ": [], "tsne wρ": [], "pca wsoftρ": [], "tsne wsoftρ": []}
    # dct = {"pca ρ": [],  "pca wρ": [], "pca wsoftρgpt": []}
    # dct = {"tsne ρ": [],  "tsne wρ": [], "tsne wsoftρgpt": []}
    dct = {"pca ρ": [], "tsne ρ": [], "pca wρ": [], "tsne wρ": [], "pca wsoftρ": [], "tsne wsoftρ": [], "pca wsoftρgpt": [], "tsne wsoftρgpt": []}
    # dct = {"pca ρ": [],  "pca wρ": [], "pca wsoftρgpt": []}
    # dct = {"tsne ρ": [],  "tsne wρ": [], "tsne wsoftρ": [], "tsne wsoftρgpt": []}
    dct = {"tsne w$\\tau$": [], "tsne $\\tau$": [], "tsne ρ": [], "tsne wρ": [], "tsne wsoftρ": [], "tsne wsoftρgpt": []}
    for s in range(1, n):
        lst = array(list(([1 / (1 + r) for r in range(s)])))
        lst /= sum(lst)
        w = tensor([lst], dtype=torch.float)
        wl = lst
        # dct["pca ρ"].append(spearmanr(ddd[0, :s], dddp[0, :s])[0])
        dct["tsne ρ"].append(spearmanr(ddd[0, :s], dddt[0, :s])[0])
        dct["tsne $\\tau$"].append(kendalltau(ddd[0, :s], dddt[0, :s])[0])
        # dct["tsne $\\tau$"].append(kendalltau(X[:s, 0], Xt[:s, 1])[0])
        # dct["pca w$\\tau$"].append(weightedtau(ddd[0, :s], dddp[0, :s], weigher=lambda r: wl[r], rank=False)[0])
        dct["tsne w$\\tau$"].append(weightedtau(ddd[0, :s], dddt[0, :s], rank=False)[0])
        # dct["pca wρ"].append(spearman(ddd[0, :s], dddp[0, :s], wl[:s]))
        dct["tsne wρ"].append(spearman(ddd[0, :s], dddt[0, :s], wl[:s]))
        # dct["pca wsoftρ"].append(float(softrho(tensor([ddd[0, :s]]), tensor([dddp[0, :s]]), w[:, :s], regularization_strength=.1)))
        dct["tsne wsoftρ"].append(float(softrho(tensor([ddd[0, :s]]), tensor([dddt[0, :s]]), w[:, :s], regularization_strength=.001)))
        # dct["pca wsoftρgpt"].append(float(spearmangpt(tensor(ddd[0, :s]), tensor(dddp[0, :s]), w[0, :s])))
        dct["tsne wsoftρgpt"].append(float(spearmangpt(tensor(ddd[0, :s]), tensor(dddt[0, :s]), w[0, :s])))
        df = DataFrame(data=dct)
    return d, dp, dt, df


# with sopen("duckdb:///:memory:") as db:
with sopen("duckdb:////home/davi/.sortedness.db") as db:
    d = hdict(n=n) >> apply(get)("d", "dp", "dt", "df") >> cache(db)
    # d >>= {("d", "dp", "dt", "df"): get(n)}
    d, dp, dt, df = d.d, d.dp, d.dt, d.df

# bins = linspace(0, 1, num=100)
# plt.hist(d, bins, alpha=0.5, label="original distances", edgecolor="k")
# plt.hist(dp, bins, alpha=0.5, label="PCA distances", edgecolor="k")
# plt.hist(dt, bins, alpha=0.5, label="TSNE distances", edgecolor="k")
# plt.legend(loc='upper right')

_, ax = plt.subplots()
style = ["--", ".", "+"]
df.plot(ax=ax, grid=True, logx=not True, style=style)
plt.show()
