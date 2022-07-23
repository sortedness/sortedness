import shelve
from dataclasses import dataclass
from functools import partial

import matplotlib.pyplot as plt
from openml.datasets import get_dataset
from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from evaluation.plot import Plot, colors
from hoshmap import idict
from hoshmap.serialization.parsing import f2hosh
from sortedness import sortedness, rank_by_distances
from sortedness.kruskal import kruskal
# warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
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
                 "wtau-sortedness": weightedtau,
                 "wtau-sortedness2": partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2))}
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
