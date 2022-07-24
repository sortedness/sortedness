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

import shelve
from functools import partial

import matplotlib.pyplot as plt
from openml.datasets import get_dataset
from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sortedness.rank import rank_by_distances

from sortedness.evaluation.plot import Plot, colors
from hoshmap import idict
from sortedness.kruskal import kruskal
from sortedness.local import ushaped_decay_f, sortedness
from sortedness.trustworthiness import trustworthiness, continuity

popular_methods = ["-1*kruskal", "trustworthiness", "continuity", "-1*nonmetric_kruskal"]
projectors = {"PCA": PCA(n_components=2).fit_transform,
              "TSNE": lambda seed, X: TSNE(n_components=2, random_state=seed).fit_transform(X)}
dataset = "abalone"
d = idict(dataset=dataset, seed=0)


def fetch(dataset):
    X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
    print("loaded", dataset)
    X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
    if dataset == "abalone":
        X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
    return X.to_numpy()


for projection, project in projectors.items():
    with shelve.open("/tmp/sortedness-cache.db") as db:
        d["X"] = fetch
        d["X_"] = project
        coefs = {"ρ-sortedness": spearmanr,
                 "tau-sortedness": kendalltau,
                 "wtau-sortedness_x": weightedtau,
                 "wtau-sortedness_x²": partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)),
                 "wtau-sortedness_Ushaped": partial(weightedtau, weigher=ushaped_decay_f(n=len(d.X)))
                 }
        f = lambda X, X_, coef: sortedness(X, X_, f=coefs[coef])
        for coef in coefs:
            d["coef"] = coef
            d[coef] = f
        d = d >> {"-1*kruskal": lambda X, X_: -kruskal(X, X_),
                  "-1*nonmetric_kruskal": lambda X, X_: -kruskal(X, X_, f=rank_by_distances),
                  "trustworthiness": trustworthiness,
                  "continuity": continuity} >> [db]
        d.show()

        # Plot
        for xlabel in popular_methods:
            p = Plot(d, projection, xlabel, "'sortedness'", legend=True, plt=plt)
            for slabel, color in zip(coefs.keys(), colors):
                p << (slabel, color)
            p.finish()

        for xlabel in ["-1*kruskal", "-1*nonmetric_kruskal"]:
            p = Plot(d, projection, xlabel, "cont, trust", legend=True, plt=plt)
            for slabel, color in zip(popular_methods[1:-1], colors[1:]):
                p << (slabel, color)
            p.finish()

        p = Plot(d, projection, "trustworthiness", "continuity", legend=False, plt=plt)
        p << ("continuity", "blue")
        p.finish()

        p = Plot(d, projection, "-1*kruskal", "-1*nonmetric_kruskal", legend=False, plt=plt)
        p << ("-1*nonmetric_kruskal", "green")
        p.finish()

        plt.show()
