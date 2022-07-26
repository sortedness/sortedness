#  Copyright (c) 2022. Davi Pereira dos Santos
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
from functools import partial

import matplotlib.pyplot as plt
from openml.datasets import get_dataset
from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from hoshmap import idict
from shelchemy.core import sopen
from sortedness.config import local_cache_uri, remote_cache_uri
from sortedness.evaluation.plot import Plot, colors
from sortedness.kruskal import kruskal
from sortedness.local import ushaped_decay_f, sortedness, sortedness_, asortedness_, asortedness__
from sortedness.rank import rank_by_distances
from sortedness.trustworthiness import trustworthiness, continuity

functions = {"-1*kruskal": lambda X, X_: -kruskal(X, X_),
             "-1*nonmetric_kruskal": lambda X, X_: -kruskal(X, X_, f=rank_by_distances),
             "trustworthiness": trustworthiness,
             "continuity": continuity,
             # "sortedness_lw": lambda X, X_: sortedness_(X, X_, f="lw", normalized=True),
             # "asortedness_1": asortedness_,
             # "asortedness_2": asortedness__
             }
projectors = {"PCA": PCA(n_components=2).fit_transform,
              "TSNE": lambda seed, X: TSNE(n_components=2, random_state=seed).fit_transform(X)}


def fetch(dataset):
    X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
    print("loaded", dataset)
    X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
    if dataset == "abalone":
        X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
    return X.to_numpy()


def prepare_experiment(d, cache):
    corr_funcs = {"ρ-sortedness": spearmanr,
                  "tau-sortedness": kendalltau,
                  "wtau-sortedness_x": weightedtau,
                  "wtau-sortedness_x²": partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)),
                  "wtau-sortedness_Ushaped": lambda X, X_: weightedtau(X, X_, weigher=ushaped_decay_f(n=len(X)))
                  }
    f = lambda X, X_, coefname: sortedness(X, X_, f=corr_funcs[coefname])
    for coef in corr_funcs:
        d["coefname"] = coef
        d[coef] = f
    d >>= functions
    return d, corr_funcs


def plot(proj_name, d, corr_funcs):
    for xlabel in functions:
        p = Plot(d, proj_name, xlabel, "'sortedness'", legend=True, plt=plt)
        for slabel, color in zip(corr_funcs.keys(), colors):
            p << (slabel, color)
        p.finish()

    for xlabel in ["-1*kruskal", "-1*nonmetric_kruskal"]:
        p = Plot(d, proj_name, xlabel, "cont, trust", legend=True, plt=plt)
        for slabel, color in zip(list(functions.keys())[2:], colors[1:]):
            p << (slabel, color)
        p.finish()

    p = Plot(d, proj_name, "trustworthiness", "continuity", legend=False, plt=plt)
    p << ("continuity", "blue")
    p.finish()

    p = Plot(d, proj_name, "-1*kruskal", "-1*nonmetric_kruskal", legend=False, plt=plt)
    p << ("-1*nonmetric_kruskal", "green")
    p.finish()

    plt.show()


data = idict(seed=0)
for projection, project in projectors.items():
    with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
        for name in ["abalone", "iris"][:1]:
            data["dataset"] = name
            data["X"] = fetch
            data["X_"] = project
            data >>= [local, remote]
            data.evaluate()

            data, corr_functions = prepare_experiment(data, [local, remote])
            data >>= [local, remote]

            plot(projection, data, corr_functions)
            data.show()
