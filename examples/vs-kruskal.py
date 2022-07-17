from openml.datasets import get_dataset
from pandas import to_numeric
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from scipy.stats import weightedtau, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from robustress.kruskal import kruskal
from robustress.sortedness import sortedness

# warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

X = get_dataset("abalone").get_data(dataset_format="dataframe")[0]
print("loaded")
X.drop(X.columns[len(X.columns) - 1], axis=1, inplace=True)
X.replace(['M', 'I', "F"], [-1, 0, 1], inplace=True)
X = X.to_numpy()
print(X.shape)
# X_ = PCA(n_components=2).fit_transform(X)
X_ = TSNE(n_components=2).fit_transform(X)
print(X_.shape)

coefs = [spearmanr, kendalltau, weightedtau]
coefnames = ["ρ-sortedness", "𝜏-sortedness", "w𝜏-sortedness"]
ds = {}
for coefname, coef in zip(coefnames, coefs):
    print(coefname)
    ds[coefname] = sortedness(X, X_, coef)
kseries = kruskal(X, X_)

plt.title(f"Kruskal stress I versus sortedness")
ax = plt.gca()
ax.set_yscale('log')
for (name, series), color in zip(ds.items(), ["blue", "green", "red"]):
    ax.scatter(kseries, series, s=30, c=color, label=name)
plt.legend()
plt.xlabel('Kruskal Stress')
plt.ylabel('Sortedness')
plt.show()

# plt.hist(ds.values(), bins=bins, log=True, histtype="stepfilled", color=colors[:n], label=list(ds.keys()))
# plt.legend()
# plt.show()
