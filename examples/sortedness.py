# Sortedness

import numpy as np
from numpy.random import permutation
from sklearn.decomposition import PCA

from sortedness.local import sortedness

mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

s = sortedness(original, original)
print(min(s), max(s), s)
# ...

s = sortedness(original, projected2)
print(min(s), max(s), s)
# ...

s = sortedness(original, projected1)
print(min(s), max(s), s)
# ...

s = sortedness(original, projectedrnd)
# ...
