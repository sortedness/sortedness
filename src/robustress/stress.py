from functools import partial

from numpy import eye, argsort
from numpy.random import shuffle, permutation
from scipy.stats import spearmanr, weightedtau
from sklearn.decomposition import PCA

from robustress.rank import rank_by_distances, rdist_by_index_lw, rdist_by_index_iw, euclidean__n_vs_1


# noinspection PyTypeChecker
def stress(ma, mb, normalized=False, f=spearmanr, decay=None):
    """
    Calculate the rank-based stress value for each point

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected, f="iw")
    [2, 0, 2]
    >>> stress(original, projected, normalized=True, f="iw")
    [0.5, 0, 0.5]
    >>> stress(original, projected, f="lw")
    [0.6666666667, 0, 0.6666666667]
    >>> stress(original, projected, normalized=True, f="lw")
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
    >>> s = stress(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, f="lw")
    >>> s
    [3.7579365079, 3.1936507937, 1.3579365079, 0.5, 1.6730880231, 1.7023809524, 2.119047619, 0.9, 1.1675324675, 1.0151515152, 3.6365800866, 1.4301587302]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (0.0413788162, 0.3109979281, [0.3109979281, 0.2642989784, 0.1123796103, 0.0413788162, 0.1384608036, 0.140885017, 0.1753673639, 0.0744818692, 0.0966222228, 0.0840115359, 0.300954758, 0.1183565505])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected, f="lw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (0.3998758038, 0.7363578282, [0.4617469862, 0.6736327973, 0.4112982678, 0.5757029323, 0.5520578945, 0.3998758038, 0.5810230659, 0.4813795327, 0.7363578282, 0.5244720171, 0.5173785057, 0.568609421])

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected, f="iw")
    [2, 0, 2]
    >>> stress(original, projected, normalized=True, f="iw")
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
    >>> s = stress(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.4722222222, 0.8333333333, [0.5, 0.75, 0.4722222222, 0.6388888889, 0.6111111111, 0.4722222222, 0.6388888889, 0.5, 0.8333333333, 0.5833333333, 0.5833333333, 0.6388888889])

    >>> s = stress(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.4722222222, 0.8333333333, [0.5, 0.75, 0.4722222222, 0.6388888889, 0.6111111111, 0.4722222222, 0.6388888889, 0.5, 0.8333333333, 0.5833333333, 0.5833333333, 0.6388888889])
    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> stress(original, projected, f="iw")
    [2, 0, 2]
    >>> stress(original, projected, normalized=True, f="iw")
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
    >>> s = stress(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.0277777778, 0.2777777778, [0.2777777778, 0.25, 0.1388888889, 0.0277777778, 0.1666666667, 0.1111111111, 0.1666666667, 0.0555555556, 0.1111111111, 0.0833333333, 0.2777777778, 0.1666666667])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = stress(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (0.3998758038, 0.7363578282, [0.4617469862, 0.6736327973, 0.4112982678, 0.5757029323, 0.5520578945, 0.3998758038, 0.5810230659, 0.4813795327, 0.7363578282, 0.5244720171, 0.5173785057, 0.568609421])

    >>> s = stress(original, original)
    >>> s
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.0034965034965034336, 0.13286713286713286, [0.10839160839160839, 0.13286713286713286, 0.027972027972027858, 0.0034965034965034336, 0.06643356643356635, 0.020979020979020935, 0.0489510489510489, 0.010489510489510412, 0.0174825174825175, 0.010489510489510412, 0.10139860139860135, 0.05594405594405588])
    >>> # Weighted correlation-based stress can yield lower values as it focuses on higher ranks.
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> s = stress(original, projected, f=weightedtau)
    >>> min(s), max(s), s
    (0.006184536334150947, 0.11972552156925742, [0.10669804730557336, 0.11972552156925742, 0.057869589983841174, 0.006184536334150947, 0.0933260482905337, 0.03175545929059609, 0.06165354971460457, 0.013142139710070833, 0.057974215598516676, 0.038444100859092556, 0.10581401370278509, 0.09018727985026914])
    >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    >>> s = stress(original, projected, f=wf)
    >>> min(s), max(s), s
    (0.0012599306648203856, 0.06097693236703372, [0.04767025179771278, 0.04857347590889044, 0.03203600763905745, 0.0012599306648203856, 0.06097693236703372, 0.009929741681779292, 0.025662139911649784, 0.002908982773061508, 0.05450317824457718, 0.03538651596692205, 0.05768659251010677, 0.0554645561816447])
    >>> projected = permutation(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.36363636363636365, 0.548951048951049, [0.4230769230769231, 0.486013986013986, 0.4160839160839161, 0.48951048951048953, 0.548951048951049, 0.4370629370629371, 0.36363636363636365, 0.45454545454545453, 0.40909090909090906, 0.3986013986013986, 0.44405594405594406, 0.506993006993007])

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
        Distance criteria:
        str         =   any by_index function name: rdist_by_index_lw, rdist_by_index_iw
        callable    =   scipy correlation functions:
            weightedtau (weighted Kendall’s τ), kendalltau, spearmanr
            Meaning of resulting values for correlation-based functions:
                0.0:    perfect projection          (no stress)
                0.5:    random projection           (enough stress to erase all information)
                2.0:    worst possible projection   (theoretical; represent the opposite of the dataset)
    decay
        Decay factor to put more or less weight on near neigbors
        `decay=0` means uniform weights
        `decay=1` is meaningless as it will always result in zero stress

    Returns
    -------
        row-vector matrix with a stress value by row
    """
    result = []

    if callable(f):
        if normalized:  # pragma: no cover
            print("Warning: 'normalized=True' is ignored for correlation-based stress ('f=callable')")
        if decay is not None:  # pragma: no cover
            print("Warning: 'decay' is ignored for correlation-based stress ('f=callable')")
            print(
                "Hint: to provide weights, pass a partially applied function like 'partial(weightedtau, weigher=lambda x: 1/(x**2 + 1)"
            )
        for a, b in zip(ma, mb):
            corr = f(euclidean__n_vs_1(ma, a), euclidean__n_vs_1(mb, b))[0]
            result.append((1 - corr) / 2)
        return result

    kwargs = {} if decay is None else {"decay": decay}
    if f == "iw":
        f = rdist_by_index_iw
    elif f == "lw":
        f = rdist_by_index_lw
    else:  # pragma: no cover
        raise Exception(f"Unknown f {f}")
    for a, b in zip(ma, mb):
        ranks_ma = rank_by_distances(ma, a)
        mb_ = mb[argsort(ranks_ma)]  # Sort mb by using ma ranks.
        ranks = rank_by_distances(mb_, b)
        d = f(argsort(ranks), normalized=normalized, **kwargs)
        result.append(d)
    return result
