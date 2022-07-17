from functools import partial

from numpy import eye, argsort
from numpy.random import shuffle, permutation
from scipy.stats import spearmanr, weightedtau, kendalltau
from sklearn.decomposition import PCA

from robustress.rank import rank_by_distances, rdist_by_index_lw, rdist_by_index_iw, euclidean__n_vs_1


# noinspection PyTypeChecker
def sortedness(X_a, X_b, f=spearmanr, return_pvalues=False):
    """
    Calculate the sortedness (a anti-stress alike correlation-based measure that ignores distance proportions) value for each point
    Functions available as scipy correlation coefficients:
        ρ-sortedness (Spearman),
        𝜏-sortedness (Kendall's 𝜏),
        w𝜏-sortedness (Sebastiano Vigna weighted Kendall's 𝜏)

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = sortedness(original, original)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    >>> min(s), max(s), s
    (0.606060606060606, 0.9696969696969696, [0.6363636363636362, 0.606060606060606, 0.8484848484848484, 0.9696969696969696, 0.7575757575757575, 0.8787878787878787, 0.7878787878787877, 0.9393939393939392, 0.8787878787878787, 0.909090909090909, 0.6666666666666666, 0.7878787878787877])
    >>> pvalues
    [0.003181646992410881, 0.005380307706696595, 1.6342325370103148e-05, 5.010421677088344e-08, 0.00024002425044091711, 5.319397680508792e-06, 0.00010742344075677409, 3.2150205761316875e-07, 5.319397680508792e-06, 1.4655483405483405e-06, 0.0018032758136924804, 0.00010742344075677409]
    >>> s = sortedness(original, projected)
    >>> min(s), max(s), s
    (0.7342657342657343, 0.9930069930069931, [0.7832167832167832, 0.7342657342657343, 0.9440559440559443, 0.9930069930069931, 0.8671328671328673, 0.9580419580419581, 0.9020979020979022, 0.9790209790209792, 0.965034965034965, 0.9790209790209792, 0.7972027972027973, 0.8881118881118882])
    >>> s = sortedness(original, projected, f=weightedtau)
    >>> min(s), max(s), s
    (0.7605489568614852, 0.9876309273316981, [0.7866039053888533, 0.7605489568614852, 0.8842608200323177, 0.9876309273316981, 0.8133479034189326, 0.9364890814188078, 0.8766929005707909, 0.9737157205798583, 0.8840515688029666, 0.9231117982818149, 0.7883719725944298, 0.8196254402994617])
    >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    >>> s = sortedness(original, projected, f=wf)
    >>> min(s), max(s), s
    (0.8780461352659326, 0.9974801386703592, [0.9046594964045744, 0.9028530481822191, 0.9359279847218851, 0.9974801386703592, 0.8780461352659326, 0.9801405166364414, 0.9486757201767004, 0.994182034453877, 0.8909936435108456, 0.9292269680661559, 0.8846268149797865, 0.8890708876367106])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> s = sortedness(original, projected)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> _, pvalues = sortedness(original, projected, return_pvalues=True)
    >>> pvalues
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> _, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    >>> pvalues
    [4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09]
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = sortedness(original, projected)
    >>> min(s), max(s), s
    (-0.39860139860139865, 0.49650349650349657, [0.39860139860139865, -0.16783216783216784, 0.46153846153846156, 0.10489510489510491, 0.18881118881118883, 0.49650349650349657, 0.1258741258741259, 0.43356643356643365, -0.39860139860139865, 0.16783216783216784, 0.034965034965034975, 0.1258741258741259])
    >>> # Weighted correlation-based stress can yield lower values as it focuses on higher ranks.
    >>> s = sortedness(original, projected, f=weightedtau)
    >>> min(s), max(s), s
    (-0.4651020733837718, 0.45329734494008334, [0.3227605098543591, -0.1634548012060478, 0.37849108727150127, -0.005336963172840341, 0.13260504041824878, 0.40944653049836677, 0.09841254408278742, 0.45329734494008334, -0.4651020733837718, -0.014576778820393398, -0.14179941261700374, 0.1703981374526939])
    >>> s = sortedness(original, projected, f=wf)
    >>> min(s), max(s), s
    (-0.51719645219205, 0.5162710639809638, [0.3098681490207809, -0.15794336281630955, 0.4312618574356535, -0.055843622226620995, 0.15059539158734384, 0.4607249573077292, 0.09347400420528493, 0.5162710639809638, -0.51719645219205, -0.12129131935470096, -0.2595632212911495, 0.20448257011752424])
    >>> _, pvalues = sortedness(original, projected, f=wf, return_pvalues=True)
    >>> pvalues
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    >>> _, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    >>> pvalues
    [0.15259045898802842, 0.5452047007776174, 0.11595087782587783, 0.7373055246579552, 0.6383612539081289, 0.11595087782587783, 0.7373055246579552, 0.11595087782587783, 0.11595087782587783, 0.5452047007776174, 1.0, 0.5452047007776174]
    >>> projected = permutation(original)
    >>> s = sortedness(original, projected, f=wf)
    >>> min(s), max(s), s
    (-0.314701609990475, 0.23829059925032403, [-0.29851544202405506, -0.26182285121352383, 0.12709476627417826, -0.1198045732618854, -0.004255962101312155, -0.09676162002121147, 0.23829059925032403, -0.1825003551612558, -0.15805497961328627, 0.204719128528062, 0.002494216275595984, -0.314701609990475])
    >>> corrs, pvalues = sortedness(original, projected, return_pvalues=True)
    >>> corrs
    [0.15384615384615385, 0.027972027972027972, 0.16783216783216784, 0.02097902097902098, -0.0979020979020979, 0.1258741258741259, 0.27272727272727276, 0.09090909090909093, 0.18181818181818185, 0.2027972027972028, 0.11188811188811189, -0.013986013986013986]
    >>> pvalues
    [0.6330906812462618, 0.9312343512018808, 0.602099427786538, 0.9484022252365223, 0.762121655996299, 0.6966831093957659, 0.3910967709418962, 0.7787253962454419, 0.5717012385276553, 0.5273023541661082, 0.729194990751066, 0.9655902689187795]

    Parameters
    ----------
    X_a
        matrix with an instance by row in a given space (often the original one)
    X_b
        matrix with an instance by row in another given space (often the projected one)
    f
        Distance criteria:
        str         =   any by_index function name: rdist_by_index_lw, rdist_by_index_iw
        callable    =   scipy correlation functions:
            weightedtau (weighted Kendall’s τ), kendalltau, spearmanr
            Meaning of resulting values for correlation-based functions:
                1.0:    perfect projection          (regarding order of examples)
                0.0:    random projection           (enough distortion to have no information left when considering the overall ordering)
               -1.0:    worst possible projection   (mostly theoretical; it represents the "opposite" of the original ordering)
    return_pvalues
        For scipy correlation functions, return a tuple '«corr, pvalue»' instead of just 'corr'
        This makes more sense for Kendall's tau. [the weighted version might not have yet a established pvalue calculation method at this moment]
        The null hypothesis is that the projection is random, i.e., sortedness = 0.5.

    Returns
    -------
        list of sortedness values (or tuples that also include pvalues)
    """
    result, pvalues = [], []
    for a, b in zip(X_a, X_b):
        corr, pvalue = f(euclidean__n_vs_1(X_a, a), euclidean__n_vs_1(X_b, b))
        result.append(corr)
        pvalues.append(pvalue)
    if return_pvalues:
        return result, pvalues
    return result


def sortedness_(X_a, X_b, f="lw", normalized=False, decay=None):  # pragma: no cover
    """Implement a version of sortedness able to use other (non-)standard correlation functions.

    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> sortedness_(original, projected, f="iw")
    [2, 0, 2]
    >>> sortedness_(original, projected, normalized=True, f="iw")
    [0.0, 0, 0.0]
    >>> sortedness_(original, projected, f="lw")
    [0.6666666667, 0, 0.6666666667]
    >>> sortedness_(original, projected, normalized=True, f="lw")
    [0.2, 0, 0.2]
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
    >>> s = sortedness_(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, f="lw")
    >>> s
    [3.7579365079, 3.1936507937, 1.3579365079, 0.5, 1.6730880231, 1.7023809524, 2.119047619, 0.9, 1.1675324675, 1.0151515152, 3.6365800866, 1.4301587302]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (0.3780041439, 0.9172423676, [0.3780041439, 0.4714020433, 0.7752407793, 0.9172423676, 0.7230783929, 0.7182299659, 0.6492652722, 0.8510362617, 0.8067555545, 0.8319769282, 0.3980904841, 0.7632868991])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness_(original, projected, f="lw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = sortedness_(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (-0.4727156565, 0.2002483923, [0.0765060277, -0.3472655947, 0.1774034644, -0.1514058647, -0.104115789, 0.2002483923, -0.1620461317, 0.0372409346, -0.4727156565, -0.0489440341, -0.0347570114, -0.137218842])
    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> sortedness_(original, projected, f="iw")
    [2, 0, 2]
    >>> sortedness_(original, projected, normalized=True, f="iw")
    [0.0, 0, 0.0]
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
    >>> s = sortedness_(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.44444444439999997, 0.9444444444, [0.44444444439999997, 0.5, 0.7222222222, 0.9444444444, 0.6666666666000001, 0.7777777778, 0.6666666666000001, 0.8888888887999999, 0.7777777778, 0.8333333333999999, 0.44444444439999997, 0.6666666666000001])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = sortedness_(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (-0.6666666666000001, 0.05555555560000003, [0.0, -0.5, 0.05555555560000003, -0.2777777777999999, -0.2222222222000001, 0.05555555560000003, -0.2777777777999999, 0.0, -0.6666666666000001, -0.16666666660000007, -0.16666666660000007, -0.2777777777999999])
    >>> s = sortedness_(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.44444444439999997, 0.9444444444, [0.44444444439999997, 0.5, 0.7222222222, 0.9444444444, 0.6666666666000001, 0.7777777778, 0.6666666666000001, 0.8888888887999999, 0.7777777778, 0.8333333333999999, 0.44444444439999997, 0.6666666666000001])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = sortedness_(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (-0.6666666666000001, 0.05555555560000003, [0.0, -0.5, 0.05555555560000003, -0.2777777777999999, -0.2222222222000001, 0.05555555560000003, -0.2777777777999999, 0.0, -0.6666666666000001, -0.16666666660000007, -0.16666666660000007, -0.2777777777999999])
    >>> import numpy as np
    >>> original = np.array([[1,2,-1], [-1,3,0], [0,1,2]])
    >>> projected = np.array([[1,2], [-1,3], [0,1]])
    >>> sortedness_(original, projected, f="iw")
    [2, 0, 2]
    >>> sortedness_(original, projected, normalized=True, f="iw")
    [0.0, 0, 0.0]
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
    >>> s = sortedness_(original, original, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> s
    [20, 18, 10, 2, 12, 8, 12, 4, 8, 6, 20, 12]
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected, normalized=True, f="iw")
    >>> min(s), max(s), s
    (0.44444444439999997, 0.9444444444, [0.44444444439999997, 0.5, 0.7222222222, 0.9444444444, 0.6666666666000001, 0.7777777778, 0.6666666666000001, 0.8888888887999999, 0.7777777778, 0.8333333333999999, 0.44444444439999997, 0.6666666666000001])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness_(original, projected, f="iw")
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = sortedness_(original, projected, normalized=True, f="lw")
    >>> min(s), max(s), s
    (-0.4727156565, 0.2002483923, [0.0765060277, -0.3472655947, 0.1774034644, -0.1514058647, -0.104115789, 0.2002483923, -0.1620461317, 0.0372409346, -0.4727156565, -0.0489440341, -0.0347570114, -0.137218842])

    >>> s = sortedness_(original, original)
    >>> s
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = sortedness_(original, projected)
    >>> min(s), max(s), s
    (0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = sortedness_(original, projected)
    >>> min(s), max(s), s
    (0.5, 3.7579365079, [3.7579365079, 3.1936507937, 1.3579365079, 0.5, 1.6730880231, 1.7023809524, 2.119047619, 0.9, 1.1675324675, 1.0151515152, 3.6365800866, 1.4301587302])

    Parameters
    ----------
    X_a
        matrix with an instance by row in a given space (often the original one)
    X_b
        matrix with an instance by row in another given space (often the projected one)
    normalized
        Whether to normalize result to [0; 1] interval
        If True, divide value by the longest possible distance.
        This makes the measure dependent on the dataset size
    f
        Distance criteria:
        str         =   any by_index function name: rdist_by_index_lw, rdist_by_index_iw
    decay
        Decay factor to put more or less weight on near neigbors
        `decay=0` means uniform weights
        `decay=1` is meaningless as it will always result in zero stress

    Returns
    -------
        list of sortedness_ values
    """
    result = []
    kwargs = {} if decay is None else {"decay": decay}
    if f == "iw":
        f = rdist_by_index_iw
    elif f == "lw":
        f = rdist_by_index_lw
    else:  # pragma: no cover
        raise Exception(f"Unknown f {f}")
    for a, b in zip(X_a, X_b):
        ranks_ma = rank_by_distances(X_a, a)
        mb_ = X_b[argsort(ranks_ma)]  # Sort mb by using ma ranks.
        ranks = rank_by_distances(mb_, b)
        d = f(argsort(ranks), normalized=normalized, **kwargs)
        result.append(d)
    return result
