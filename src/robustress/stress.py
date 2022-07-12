from numpy import eye, argsort
from numpy.random import shuffle, permutation
from sklearn.decomposition import PCA

from robustress.rank import rank_by_distances, rdist_by_index_lw, rdist_by_index_iw


# noinspection PyTypeChecker
def stress(ma, mb, normalized=False, f: callable = rdist_by_index_iw, decay=None):
    """
    Calculate the rank-based stress value for each point

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected)
    [2, 0, 2]
    >>> stress(original, projected, normalized=True)
    [0.5, 0, 0.5]
    >>> stress(original, projected, f=rdist_by_index_lw)
    [0.6666666667, 0, 0.6666666667]
    >>> stress(original, projected, normalized=True, f=rdist_by_index_lw)
    [0.4, 0, 0.4]
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> original
    array([[ 1.12573022,  1.86789514],
           [ 1.64042265,  2.10490012],
           [ 0.46433063,  2.36159505],
           [ 2.30400005,  2.94708096],
           [ 0.29626476,  0.73457853],
           [ 0.37672554,  2.04132598],
           [-1.32503077,  1.78120834],
           [-0.24591095,  1.26773265],
           [ 0.45574102,  1.68369984],
           [ 1.41163054,  3.04251337],
           [ 0.87146534,  3.36646347],
           [ 0.33480533,  2.35151007]])
    >>> s = stress(original, original)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, f=rdist_by_index_lw)
    >>> s
    [3.7579365079, 3.1936507937, 1.3579365079, 0.5, 1.6730880231, 1.7023809524, 2.119047619, 0.9, 1.1675324675, 1.0151515152, 3.6365800866, 1.4301587302]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True, f=rdist_by_index_lw)
    >>> min(s), max(s), s
    (0.0413788162, 0.3109979281, [0.3109979281, 0.2642989784, 0.1123796103, 0.0413788162, 0.1384608036, 0.140885017, 0.1753673639, 0.0744818692, 0.0966222228, 0.0840115359, 0.300954758, 0.1183565505])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected, f=rdist_by_index_lw)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True, f=rdist_by_index_lw)
    >>> min(s), max(s), s
    (0.3998758038, 0.7363578282, [0.4617469862, 0.6736327973, 0.4112982678, 0.5757029323, 0.5520578945, 0.3998758038, 0.5810230659, 0.4813795327, 0.7363578282, 0.5244720171, 0.5173785057, 0.568609421])

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected)
    [2, 0, 2]
    >>> stress(original, projected, normalized=True)
    [0.5, 0, 0.5]
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> original
    array([[ 1.12573022,  1.86789514],
           [ 1.64042265,  2.10490012],
           [ 0.46433063,  2.36159505],
           [ 2.30400005,  2.94708096],
           [ 0.29626476,  0.73457853],
           [ 0.37672554,  2.04132598],
           [-1.32503077,  1.78120834],
           [-0.24591095,  1.26773265],
           [ 0.45574102,  1.68369984],
           [ 1.41163054,  3.04251337],
           [ 0.87146534,  3.36646347],
           [ 0.33480533,  2.35151007]])
    >>> s = stress(original, original)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.4722222222, 0.8333333333, [0.5, 0.75, 0.4722222222, 0.6388888889, 0.6111111111, 0.4722222222, 0.6388888889, 0.5, 0.8333333333, 0.5833333333, 0.5833333333, 0.6388888889])

    >>> s = stress(original, original)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.4722222222, 0.8333333333, [0.5, 0.75, 0.4722222222, 0.6388888889, 0.6111111111, 0.4722222222, 0.6388888889, 0.5, 0.8333333333, 0.5833333333, 0.5833333333, 0.6388888889])
    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected)
    [2, 0, 2]
    >>> stress(original, projected, normalized=True)
    [0.5, 0, 0.5]
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> original
    array([[ 1.12573022,  1.86789514],
           [ 1.64042265,  2.10490012],
           [ 0.46433063,  2.36159505],
           [ 2.30400005,  2.94708096],
           [ 0.29626476,  0.73457853],
           [ 0.37672554,  2.04132598],
           [-1.32503077,  1.78120834],
           [-0.24591095,  1.26773265],
           [ 0.45574102,  1.68369984],
           [ 1.41163054,  3.04251337],
           [ 0.87146534,  3.36646347],
           [ 0.33480533,  2.35151007]])
    >>> s = stress(original, original)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.4722222222, 0.8333333333, [0.5, 0.75, 0.4722222222, 0.6388888889, 0.6111111111, 0.4722222222, 0.6388888889, 0.5, 0.8333333333, 0.5833333333, 0.5833333333, 0.6388888889])

    Parameters
    ----------
    ma
        matrix with an instance by row in a given space (called original)
    mb
        matrix with an instance by row in another given space (called projected)
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
        This makes the measure dependent on the dataset size
    f
        Distance criteria
    decay
        Decay factor to put more or less weight on near neigbors
        `decay=0` means uniform weights
        `decay=1` is meaningless as it will always result in zero stress

    Returns
    -------
        row-vector matrix with a stress value by row
    """
    kwargs = {} if decay is None else {"decay": decay}
    result = []
    for a, b in zip(ma, mb):
        ranks_ma = rank_by_distances(ma, a)
        mb_ = mb[argsort(ranks_ma)]  # Sort mb by using ma ranks.
        ranks = rank_by_distances(mb_, b)
        d = f(argsort(ranks), normalized=normalized, **kwargs)
        result.append(d)
    return result
