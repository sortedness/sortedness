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

from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sortedness.local import sortedness, stress
from sortedness.trustworthiness import trustworthiness, continuity

functions = {
    "-1*kruskal": lambda X, X_: -stress(X, X_),
    "-1*nonmetric_kruskal": lambda X, X_: -stress(X, X_, metric=False),
    "trustworthiness": trustworthiness,
    "continuity": continuity
}
projectors = {
    "PCA": PCA(n_components=2).fit_transform,
    "TSNE": lambda seed, X: TSNE(n_components=2, random_state=seed).fit_transform(X)
}
corr_funcs = {
    "ρ": spearmanr,
    "tau": kendalltau,
    "wtau-x": weightedtau,
    "wtau-x²": lambda X, X_: weightedtau(X, X_, weigher=lambda x: 1 / (1 + x ** 2)),
    "sortedness": None,
}


# def fetch_asnumpy(dataset):
#     print(f"Loading {dataset}...", end="\t", flush=True)
#     X = get_dataset(dataset).get_data(dataset_format="dataframe")[0]
#     X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
#     if dataset == "abalone":
#         X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
#     print("loaded!")
#     return X.to_numpy()


def prepare_experiment(d):
    f = lambda X, X_, coefname: sortedness(X, X_, f=corr_funcs[coefname])
    for coefname in corr_funcs:
        d["coefname"] = coefname
        d[coefname] = f  # Apply proposed f functions (but don't evaluate yet)
    d >>= functions  # Apply all f functions from literature (but don't evaluate yet)
    return d


# def plot(proj_name, d):
#     for xlabel in functions:
#         p = Plot(d, proj_name, xlabel, "'sortedness'", legend=True, plt=plt)
#         for slabel, color in zip(corr_funcs.keys(), colors):
#             p << (slabel, color)
#         p.finish()
#
#     for xlabel in ["-1*kruskal", "-1*nonmetric_kruskal"]:
#         p = Plot(d, proj_name, xlabel, "cont, trust", legend=True, plt=plt)
#         for slabel, color in zip(list(functions.keys())[2:], colors[1:]):
#             p << (slabel, color)
#         p.finish()
#
#     p = Plot(d, proj_name, "trustworthiness", "continuity", legend=False, plt=plt)
#     p << ("continuity", "blue")
#     p.finish()
#
#     p = Plot(d, proj_name, "-1*kruskal", "-1*nonmetric_kruskal", legend=False, plt=plt)
#     p << ("-1*nonmetric_kruskal", "green")
#     p.finish()
#
#     plt.show()
#
#
# data = idict(seed=0)
# for projection, fproject in projectors.items():
#     with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
#         for name in ["abalone", "iris"]:
#             data["dataset"] = name
#             data["X"] = fetch_asnumpy  # Apply proposed fetch() (but don't evaluate yet)
#             data["X_"] = fproject  # Apply proposed fproject() (but don't evaluate yet)
#
#             data = prepare_experiment(data)
#             data >>= [local, remote]  # Add caches (but don't send/fetch anything yet)
#             data.evaluate()  # Evaluate each needed field (and store it, or just fetch it when possible)
#
#             plot(f"{name}: {projection}", data)
#             data.show()