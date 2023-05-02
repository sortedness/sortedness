#  Copyright (c) 2022. Davi Pereira dos Santos
#  This file is part of the sortedness project.
#  Please respect the license - more about this in the section (*) below.
#
#  sortedness is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sortedness is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
#
#  (*) Removing authorship by any means, e.g. by distribution of derived
#  works or verbatim, obfuscated, compiled or rewritten versions of any
#  part of this work is illegal and it is unethical regarding the effort and
#  time spent here.
#

import gc
from functools import partial

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, mean, sqrt
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata, kendalltau, weightedtau
from shelchemy.lazy import ichunks

from sortedness.parallel import rank_alongrow, rank_alongcol
from sortedness.wtau import parwtau


def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


weightedtau.isweightedtau = True


def sortedness(X, X_, f=weightedtau, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
     Calculate the sortedness (stress-like correlation-based measure that ignores distance proportions) value for each point
     Functions available as scipy correlation coefficients:
         ρ-sortedness (Spearman),
         𝜏-sortedness (Kendall's 𝜏),
         w𝜏-sortedness (Sebastiano Vigna weighted Kendall's 𝜏)  ← default

    # TODO?: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        This probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        Values can be lower due to the presence of ties, but only when the projection isn't perfect for all points.
        In the end, it might be even desired to penalize ties, as they don't exactly contribute to a stronger ordering and are (probabilistically) easier to be kept than a specific order.

    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    f
        Distance criteria:
        callable    =   scipy correlation function:
            weightedtau (weighted Kendall’s τ is the default), kendalltau, spearmanr
            Meaning of resulting values for correlation-based functions:
                1.0:    perfect projection          (regarding order of examples)
                0.0:    random projection           (enough distortion to have no information left when considering the overall ordering)
               -1.0:    worst possible projection   (mostly theoretical; it represents the "opposite" of the original ordering)
    return_pvalues
        For scipy correlation functions, return a 2-column matrix 'corr, pvalue' instead of just 'corr'
        This makes more sense for Kendall's tau. [the weighted version might not have yet a established pvalue calculation method at this moment]
        The null hypothesis is that the projection is random, i.e., sortedness = 0.0.
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    parallel_kwargs
        Any extra argument to be provided to pathos parallelization
    parallel_n_trigger
        Threshold to disable parallelization for small n values
    kwargs
        Arguments to be passed to the correlation measure

     Returns
     -------
         list of sortedness values (or tuples that also include pvalues)


    >>> ll = [[i] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> r = sortedness(a, b)
    >>> min(r), max(r)
    (-1.0, 0.998638259786)

    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> r = sortedness(a, b)
    >>> r
    array([ 0.19597951, -0.37003858,  0.06014087, -0.42174564,  0.2448619 ,
            0.24635858,  0.16814336,  0.06919001,  0.1627886 ,  0.33136454,
            0.41592274, -0.10615388,  0.17549727,  0.17371479, -0.21360864,
           -0.3677769 , -0.08292823])
    >>> min(r), max(r)
    (-0.421745643027, 0.415922739891)
    >>> round(mean(r), 12)
    0.040100605153

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> from sklearn.decomposition import PCA
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)

    >>> s = sortedness(original, original)
    >>> min(s), max(s), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> s = sortedness(original, projected2)
    >>> min(s), max(s), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> s = sortedness(original, projected1)
    >>> min(s), max(s), s
    (0.432937128932, 0.944810120534, array([0.43293713, 0.53333015, 0.88412753, 0.94481012, 0.81485109,
           0.81330052, 0.76691474, 0.91169619, 0.88998817, 0.90102615,
           0.61372341, 0.86996213]))
    >>> s = sortedness(original, projectedrnd)
    >>> min(s), max(s), s
    (-0.578096068617, 0.396112816715, array([ 0.21296126, -0.57809607,  0.33083346, -0.00638865, -0.16007932,
            0.39611282, -0.27357934,  0.04360717, -0.54534052,  0.19042181,
           -0.32805008, -0.04178184]))

    >>> sortedness(original, original, f=kendalltau, return_pvalues=True)
    array([[1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08],
           [1.0000e+00, 5.0104e-08]])
    >>> sortedness(original, projected2, f=kendalltau)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> sortedness(original, projected1, f=kendalltau)
    array([0.56363636, 0.52727273, 0.81818182, 0.96363636, 0.70909091,
           0.85454545, 0.74545455, 0.92727273, 0.85454545, 0.89090909,
           0.6       , 0.74545455])
    >>> sortedness(original, projectedrnd, f=kendalltau)
    array([ 0.2       , -0.38181818,  0.23636364, -0.09090909, -0.05454545,
            0.23636364, -0.09090909,  0.23636364, -0.63636364, -0.01818182,
           -0.2       , -0.01818182])

    >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    >>> sortedness(original, original, f=wf, return_pvalues=True)
    array([[ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan],
           [ 1., nan]])
    >>> sortedness(original, projected2, f=wf)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> sortedness(original, projected1, f=wf)
    array([0.37051697, 0.32224383, 0.94418603, 0.92935689, 0.88533575,
           0.75100956, 0.75075359, 0.88898941, 0.91311278, 0.91054531,
           0.53138984, 0.95437837])
    >>> sortedness(original, projectedrnd, f=wf)
    array([ 0.05789726, -0.58800145,  0.24738378, -0.09877965, -0.1727096 ,
            0.53783509, -0.302085  ,  0.03559481, -0.65891558, -0.00798687,
           -0.47395526, -0.1452215 ])
    >>> np.random.seed(14980)
    >>> projectedrnd = permutation(original)
    >>> sortedness(original, projectedrnd)
    array([ 0.21666209, -0.23798067,  0.16830046, -0.5559795 , -0.23470751,
            0.35716692, -0.0054951 , -0.35890385,  0.13101743, -0.48460059,
           -0.46430457,  0.27400939])
    >>> sortedness(original, np.flipud(original))
    array([-0.26471073,  0.39699442,  0.05022757,  0.21215372,  0.23467884,
           -0.33277347, -0.31616633,  0.19381204, -0.16114967,  0.08829425,
            0.3383928 , -0.31012412])
    >>> original = np.array([[0],[1],[2],[3],[4],[5],[6]])
    >>> projected = np.array([[6],[5],[4],[3],[2],[1],[0]])
    >>> sortedness(original, projected)
    array([1., 1., 1., 1., 1., 1., 1.])
    >>> projected = np.array([[0],[6],[5],[4],[3],[2],[1]])
    >>> sortedness(original, projected)
    array([-1.        ,  0.42263889,  0.80668827,  0.98180162,  0.98180162,
            0.82721863,  0.61648537])
    """
    if hasattr(f, "isweightedtau") and f.isweightedtau and "rank" not in kwargs:
        kwargs["rank"] = None
    if parallel_kwargs is None:
        parallel_kwargs = {}
    result, pvalues = [], []
    npoints = len(X)

    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    sqdist_X, sqdist_X_ = tmap(lambda M: cdist(M, M), [X, X_])

    # For f=weightedtau: scores = -ranks.
    scores_X = -remove_diagonal(sqdist_X)
    scores_X_ = -remove_diagonal(sqdist_X_)

    # TODO: Offer option to use a parallel wtau (when/if it becomes a working thing)
    for i in range(len(X)):
        corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
        result.append(round(corr, 12))
        pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def pwsortedness(X, X_, parallel=True, parallel_n_trigger=200, batches=10, debug=False, dist=None, **parallel_kwargs):
    """
    Local pairwise sortedness (Λ𝜏w) based on Sebastiano Vigna weighted Kendall's 𝜏

    Importance rankings are calculated internally based on proximity of each pair to the point of interest.

    # TODO?: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        This probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        Values can be lower due to the presence of ties, but only when the projection isn't prefect for all points.
        In the end, it might be even desired to penalize ties, as they don't exactly contribute to a stronger ordering and are (probabilistically) easier to be kept than a specific order.

    Parameters
    ----------
    X
        Original dataset
    X_
        Projected points
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    parallel_kwargs
        Any extra argument to be provided to pathos parallelization
    parallel_n_trigger
        Threshold to disable parallelization for small n values
    batches
        Parallel batch size
    debug
        Whether to print more info
    dist
        Provide distance matrices (D, D_) instead of points
        X and X_ are ignored

    Returns
    -------
        Numpy vector

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> m = (1, 12)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(m, cov, size=12)
    >>> from sklearn.decomposition import PCA
    >>> projected2 = PCA(n_components=2).fit_transform(original)
    >>> projected1 = PCA(n_components=1).fit_transform(original)
    >>> np.random.seed(0)
    >>> projectedrnd = permutation(original)

    >>> r = pwsortedness(original, original)
    >>> min(r), max(r), round(mean(r), 12)
    (1.0, 1.0, 1.0)
    >>> r = pwsortedness(original, projected2)
    >>> min(r), round(mean(r), 12), max(r)
    (1.0, 1.0, 1.0)
    >>> r = pwsortedness(original, projected1)
    >>> min(r), round(mean(r), 12), max(r)
    (0.730078995423, 0.774457348878, 0.837310352695)
    >>> r = pwsortedness(original, projected2[:, 1:])
    >>> min(r), round(mean(r), 12), max(r)
    (0.18672441995, 0.281712132364, 0.420214364305)
    >>> r = pwsortedness(original, projectedrnd)
    >>> min(r), round(mean(r), 12), max(r)
    (-0.198780473657, -0.064598420372, 0.147224384381)
    """
    npoints = len(X) if X is not None else len(dist[0])  # pragma: no cover
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    if debug:  # pragma: no cover
        print(1)
    thread = lambda M: -pdist(M, metric="sqeuclidean")
    scores_X, scores_X_ = tmap(thread, [X, X_]) if X is not None else (-squareform(dist[0]), -squareform(dist[1]))
    if debug:  # pragma: no cover
        print(2)
    D, D_ = tmap(squareform, [-scores_X, -scores_X_]) if dist is None else (dist[0], dist[1])
    if debug:  # pragma: no cover
        print(3)
    n = len(D)
    m = (n**2 - n) // 2

    def makeM(E):
        M = np.zeros((m, n))
        if debug:  # pragma: no cover
            print(4)
        c = 0
        for i in range(n - 1):  # a bit slow, but only a fraction of wtau (~5%)
            h = n - i - 1
            d = c + h
            M[c:d] = E[i] + E[i + 1 :]
            c = d
        if debug:  # pragma: no cover
            print(5)
        del E
        gc.collect()
        return M.T

    M = makeM(D)
    if debug:  # pragma: no cover
        print(6)
    R = rank_alongrow(M, step=n // batches, parallel=parallel, **parallel_kwargs).T
    del M
    gc.collect()
    if debug:  # pragma: no cover
        print(7)
    res = np.round(parwtau(scores_X, scores_X_, npoints, R, parallel=parallel, **parallel_kwargs), 12)
    del R
    gc.collect()

    # M_ = makeM(D_)
    # if debug: print(6)
    # R_ = rank_alongrow(M_, step=n // batches, parallel=parallel, **parallel_kwargs).T
    # del M_
    # gc.collect()
    # if debug: print(7)
    # r_ = np.round(parwtau(scores_X, scores_X_, npoints, parallel=parallel, **parallel_kwargs), 12)
    # del R_
    # gc.collect()
    return res


def rsortedness(X, X_, f=weightedtau, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
    Reciprocal sortedness: consider the neighborhood relation the other way around

    Might be good to assess the effect of a projection on hubness, and also to serve as a loss function for a custom projection algorithm.

    WARNING: this function is experimental, i.e., not as well tested as the others; it might need a better algorithm/fomula as well.

    # TODO?: add flag to break (not so rare) cases of ties that persist after projection (implies a much slower algorithm)
        This probably doesn't make any difference on the result, except on categorical, pathological or toy datasets
        Values can be lower due to the presence of ties, but only when the projection isn't prefect for all points.
        In the end, it might be even desired to penalize ties, as they don't exactly contribute to a stronger ordering and are (probabilistically) easier to be kept than a specific order.

    Parameters
    ----------
    f
        Distance criteria:
        callable    =   scipy correlation function:
            weightedtau (weighted Kendall’s τ is the default), kendalltau, spearmanr
            Meaning of resulting values for correlation-based functions:
                1.0:    perfect projection          (regarding order of examples)
                0.0:    random projection           (enough distortion to have no information left when considering the overall ordering)
               -1.0:    worst possible projection   (mostly theoretical; it represents the "opposite" of the original ordering)
    return_pvalues
        For scipy correlation functions, return a 2-column matrix 'corr, pvalue' instead of just 'corr'
        This makes more sense for Kendall's tau. [the weighted version might not have yet a established pvalue calculation method at this moment]
        The null hypothesis is that the projection is random, i.e., sortedness = 0.0.
    parallel
        None: Avoid high-memory parallelization
        True: Full parallelism
        False: No parallelism
    parallel_kwargs
        Any extra argument to be provided to pathos parallelization
    parallel_n_trigger
        Threshold to disable parallelization for small n values
    kwargs
        Arguments to be passed to the correlation measure

    Returns
    -------
        Numpy vector


    >>> ll = [[i, ] for i in range(17)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> #r = rsortedness(a, b)
    >>> #min(r), max(r)
    (-0.707870893072, 0.962964134515)

    >>> rnd = np.random.default_rng(0)
    >>> rnd.shuffle(ll)
    >>> b = np.array(ll)
    >>> b.ravel()
    array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    >>> r = rsortedness(b, a)
    >>> r
    array([ 0.36861667, -0.07147685,  0.39350142, -0.04581926, -0.03951645,
            0.31100414, -0.18107755,  0.28268222, -0.29248869,  0.19177107,
            0.48076521, -0.17640674,  0.13098522,  0.34833996,  0.01844146,
           -0.58291518,  0.34742337])
    >>> min(r), max(r), round(mean(r), 12)
    (-0.582915181328, 0.480765206133, 0.087284118913)
    >>> rsortedness(b, a, f=kendalltau, return_pvalues=True)
    array([[ 0.2316945 ,  0.04863508],
           [-0.37005403,  0.21347624],
           [ 0.17618709,  0.04863508],
           [-0.35418588,  0.21347624]])
    """
    if hasattr(f, "isweightedtau") and f.isweightedtau and "rank" not in kwargs:
        kwargs["rank"] = None
    if parallel_kwargs is None:
        parallel_kwargs = {}
    npoints = len(X)
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    D, D_ = tmap(lambda M: cdist(M, M, metric="sqeuclidean"), [X, X_])
    R, R_ = (rank_alongcol(M, parallel=parallel) for M in [D, D_])
    scores_X, scores_X_ = tmap(lambda M: -remove_diagonal(M), [R, R_])  # For f=weightedtau: scores = -ranks.

    if hasattr(f, "isparwtau"):  # pragma: no cover
        raise Exception("TODO: Pairtau implementation disagree with scipy weightedtau")
        # return parwtau(scores_X, scores_X_, npoints, parallel=parallel, **kwargs)

    def thread(l):
        lst1 = []
        lst2 = []
        for i in l:
            corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
            lst1.append(round(corr, 12))
            lst2.append(round(pvalue, 12))
        return lst1, lst2

    result, pvalues = [], []
    jobs = pmap(thread, ichunks(range(npoints), 15, asgenerators=False))
    for corrs, pvalues in jobs:
        result.extend(corrs)
        pvalues.extend(pvalues)

    result = np.array(result, dtype=float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def stress(X, X_, metric=True, parallel=True, parallel_size_trigger=10000, **parallel_kwargs):
    """
    Kruskal's "Stress Formula 1" normalized before comparing distances.
    default: Euclidean

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> mean = (1, 12)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(mean, cov, size=12)
    >>> s = stress(original, original*5)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> from sklearn.decomposition import PCA
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = stress(original, projected)
    >>> min(s), max(s), s
    (0.073462710191, 0.333885390367, array([0.2812499 , 0.31103416, 0.21994448, 0.07346271, 0.2810867 ,
           0.16411944, 0.17002148, 0.14748528, 0.18341208, 0.14659984,
           0.33388539, 0.24110857]))
    >>> stress(original, projected)
    array([0.2812499 , 0.31103416, 0.21994448, 0.07346271, 0.2810867 ,
           0.16411944, 0.17002148, 0.14748528, 0.18341208, 0.14659984,
           0.33388539, 0.24110857])
    >>> stress(original, projected, metric=False)
    array([0.33947258, 0.29692937, 0.30478874, 0.10509128, 0.2516135 ,
           0.2901905 , 0.1662822 , 0.13153341, 0.34299717, 0.164696  ,
           0.35266095, 0.35276684])



    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
        matrix with an instance by row in another given space (often the projected one)
    metric
        Stress formula version: metric or nonmetric
    parallel
        Parallelize processing when |X|>1000. Might use more memory.

    Returns
    -------

    """
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and X.size > parallel_size_trigger else map
    # TODO: parallelize cdist in slices?
    if metric:
        thread = lambda M, m: cdist(M, M, metric=m)
        Dsq, D_ = tmap(thread, [X, X_], ["sqeuclidean", "Euclidean"])  # Slowest part (~98%).
        Dsq /= np.max(Dsq)
        D = sqrt(Dsq)
        D_ /= np.max(D_)
    else:
        thread = lambda M: rankdata(cdist(M, M, metric="sqeuclidean"), method="average", axis=1) - 1
        D, D_ = tmap(thread, [X, X_])
        Dsq = D**2

    sqdiff = (D - D_) ** 2
    nume = sum(sqdiff)
    deno = sum(Dsq)
    result = np.round(sqrt(nume / deno), 12)
    return result
