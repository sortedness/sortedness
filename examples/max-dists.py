from numpy import eye
from numpy.linalg import norm
from numpy.random import randint, shuffle

from robustress.rank import rdist_by_index_lw, stress

old = 0
for l in range(3000000000000000):
    d = rdist_by_index_lw(list(range(l - 1, -1, -1)), normalized=False)
    # dnorm = rank_based_dist__by_index(list(range(l - 1, -1, -1)))
    print(l, "\t", d, "\t", d - old)
    old = d

#
# import numpy as np
# projected = np.array([[1,2], [-1,3], [0,1]])
#
# print(norm(projected-projected[0], axis=1, keepdims=True))
#
# import numpy as np
#
# mean = (1, 2)
# cov = eye(2)
# rng = np.random.default_rng(seed=6)
# original = rng.multivariate_normal(mean, cov, size=5)
# print(stress(original, original))
#
# r = np.array(list(range(10)))
# np.random.seed(0)
# shuffle(r)
# print(r)
# print(r[np.argsort(r)])
