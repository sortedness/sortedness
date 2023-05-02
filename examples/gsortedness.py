# Sortedness

import numpy as np
from numpy.random import permutation
from sklearn.decomposition import PCA

from sortedness import global_pwsortedness

# Some synthetic data.
mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

# Print measurement result and p-value.
s = global_pwsortedness(original, original)
print(list(s))
# ...

s = global_pwsortedness(original, projected2)
print(list(s))
# ...

s = global_pwsortedness(original, projected1)
print(list(s))
# ...

s = global_pwsortedness(original, projectedrnd)
print(list(s))
# ...
