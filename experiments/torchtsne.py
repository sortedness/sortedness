import os

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = ""

n = 1797//4

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target
X, y = datax[:n], datay[:n]
X = X.astype(np.float32)
print(X.shape, y.shape)

# from sklearn.manifold import TSNE
# X_ = TSNE(random_state=42, n_components=2, verbose=0, n_jobs=-1).fit_transform(X)
from tsne_torch import TorchTSNE

X = PCA(n_components=6).fit_transform(X)
X_ = TorchTSNE(verbose=True).fit_transform(X)
DataFrame(X_)
plt.scatter(X_[:, 0], X_[:, 1], c=y)
plt.show()
