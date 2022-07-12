from numpy import eye, argsort
from numpy.random import shuffle, permutation
from sklearn.decomposition import PCA

from robustress.rank import rank_by_distances, rank_dist__by_index


def stress(ma, mb, normalized=False):
    """
    Calculate the stress for each point

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected)
    [0.6666666667, 0, 0.6666666667]
    >>> stress(original, projected, normalized=True)
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
    >>> s = stress(original, projected)
    >>> s
    [3.7579365079, 3.1936507937, 1.3579365079, 0.5, 1.6730880231, 1.7023809524, 2.119047619, 0.9, 1.1675324675, 1.0151515152, 3.6365800866, 1.4301587302]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.0413788162, 0.3109979281, [0.3109979281, 0.2642989784, 0.1123796103, 0.0413788162, 0.1384608036, 0.140885017, 0.1753673639, 0.0744818692, 0.0966222228, 0.0840115359, 0.300954758, 0.1183565505])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True)
    >>> min(s), max(s), s
    (0.3998758038, 0.7363578282, [0.4617469862, 0.6736327973, 0.4112982678, 0.5757029323, 0.5520578945, 0.3998758038, 0.5810230659, 0.4813795327, 0.7363578282, 0.5244720171, 0.5173785057, 0.568609421])

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

    Returns
    -------
        row-vector matrix with a stress value by row
    """
    result = []
    for a, b in zip(ma, mb):
        ranks_ma = rank_by_distances(
            ma,
            a,
        )
        mb_ = mb[argsort(ranks_ma)]  # Sort mb by using ma ranks.
        ranks = rank_by_distances(mb_, b)
        d = rank_dist__by_index(argsort(ranks), normalized=normalized)
        result.append(d)
    return result
