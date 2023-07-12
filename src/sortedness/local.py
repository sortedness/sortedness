#  Copyright (c) 2023. Davi Pereira dos Santos
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
from numpy import eye, mean, sqrt, ndarray
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata, kendalltau, weightedtau

from sortedness.parallel import rank_alongrow, rank_alongcol


def remove_diagonal(X):
    n_points = len(X)
    nI = ~eye(n_points, dtype=bool)  # Mask to remove diagonal.
    return X[nI].reshape(n_points, -1)


weightedtau.isweightedtau = True


def sortedness(X, X_, i=None, symmetric=True, f=weightedtau, distance_dependent=True, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
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
    i
        None:   calculate sortedness for all instances
        `int`:  index of the instance of interest
    symmetric
        True: Take the mean between extrusion and intrusion emphasis
            Equivalent to `(sortedness(a, b) + sortedness(b, a)) / 2` at a slightly lower cost.
            Might increase memory usage.
        False: Weight by original distances (extrusion emphasis), not the projected distances.
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
    array([ 0.24691868, -0.17456491,  0.19184376, -0.18193532,  0.07175694,
            0.27992254,  0.04121859,  0.16249574, -0.03506842,  0.27856259,
            0.40866965, -0.07617887,  0.12184064,  0.24762942, -0.05049511,
           -0.46277399,  0.12193493])
    >>> min(r), max(r)
    (-0.462773990559, 0.408669653064)
    >>> round(mean(r), 12)
    0.070104521222

    >>> import numpy as np
    >>> from functools import partial
    >>> from scipy.stats import spearmanr, weightedtau
    >>> me = (1, 2)
    >>> cov = eye(2)
    >>> rng = np.random.default_rng(seed=0)
    >>> original = rng.multivariate_normal(me, cov, size=12)
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
    (0.393463224666, 0.944810120534, array([0.43735232, 0.39346322, 0.88256382, 0.94481012, 0.794352  ,
           0.78933235, 0.7409755 , 0.90341771, 0.89081602, 0.90102615,
           0.53953244, 0.86131572]))
    >>> s = sortedness(original, projectedrnd)
    >>> min(s), max(s), s
    (-0.648305479567, 0.397019507592, array([ 0.12977864, -0.49887948,  0.23107955, -0.09591571, -0.12509467,
            0.39701951, -0.21772049,  0.11895569, -0.64830548,  0.00279294,
           -0.34542772, -0.09307021]))

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
    array([0.89469168, 0.89269637, 0.92922928, 0.99721669, 0.86529591,
           0.97806422, 0.94330979, 0.99357377, 0.87959707, 0.92182767,
           0.87256459, 0.87747329])
    >>> sortedness(original, projectedrnd, f=wf)
    array([ 0.23771513, -0.2790059 ,  0.3718005 , -0.16623167,  0.06179047,
            0.40434396, -0.00130294,  0.46569739, -0.67581876, -0.23852189,
           -0.39125007,  0.12131153])
    >>> np.random.seed(14980)
    >>> projectedrnd = permutation(original)
    >>> sortedness(original, projectedrnd)
    array([ 0.24432153, -0.19634576, -0.00238081, -0.4999116 , -0.01625951,
            0.22478766,  0.07176118, -0.48092843,  0.19345964, -0.44895295,
           -0.42044773,  0.06942218])
    >>> sortedness(original, np.flipud(original))
    array([-0.28741742,  0.36769361,  0.06926091,  0.02550202,  0.21424544,
           -0.3244699 , -0.3244699 ,  0.21424544,  0.02550202,  0.06926091,
            0.36769361, -0.28741742])
    >>> original = np.array([[0],[1],[2],[3],[4],[5],[6]])
    >>> projected = np.array([[6],[5],[4],[3],[2],[1],[0]])
    >>> sortedness(original, projected)
    array([1., 1., 1., 1., 1., 1., 1.])
    >>> projected = np.array([[0],[6],[5],[4],[3],[2],[1]])
    >>> sortedness(original, projected)
    array([-1.        ,  0.51956213,  0.81695345,  0.98180162,  0.98180162,
            0.81695345,  0.51956213])
    >>> sortedness(original, projected, 1)
    0.519562134793
    >>> sortedness(original, projected, 1, symmetric=False)
    0.422638894922
    >>> sortedness([[1,2,3,3],[1,2,7,3],[3,4,8,5],[1,8,3,5]], [[2,1,2,3],[3,1,2,3],[5,4,5,6],[9,7,6,3]], 1)
    0.181818181818
    """
    isweightedtau = False
    if hasattr(f, "isweightedtau") and f.isweightedtau:
        isweightedtau = True
        if not symmetric:
            if "rank" in kwargs:  # pragma: no cover
                raise Exception(f"Cannot set `symmetric=False` and provide `rank` at the same time.")
            kwargs["rank"] = None
    elif not symmetric:  # pragma: no cover
        raise Exception(f"`symmetric=False` not implemented for custom `f`")
    if parallel_kwargs is None:
        parallel_kwargs = {}
    result, pvalues = [], []
    npoints = len(X)

    if i is not None:
        x = X[i] if isinstance(X, (ndarray, list)) else X.iloc[i].to_numpy()
        x_ = X_[i] if isinstance(X_, (ndarray, list)) else X_.iloc[i].to_numpy()
        X = np.delete(X, i, axis=0)
        X_ = np.delete(X_, i, axis=0)
        d_ = np.sum((X_ - x_) ** 2, axis=1)
        if distance_dependent:
            d = np.sum((X - x) ** 2, axis=1)
            scores_X, scores_X_ = (-d, -d_) if isweightedtau else (d, d_)
            corr, pvalue = f(scores_X, scores_X_, **kwargs)
            return (np.round(corr, 12), pvalue) if return_pvalues else np.round(corr, 12)
        else:  # pragma: no cover
            raise Exception(f"Not implemented yet; it is an open problem")
            # D = abs(X - x).T
            # scores_X, scores_x_ = (-D, -d_) if isweightedtau else (D, d_)
            # for j in range(len(scores_X)):
            #     corr, pvalue = f(scores_X[j], scores_x_, **kwargs)
            #     result.append(round(corr, 12))
            #     pvalues.append(round(pvalue, 12))
            # return (mean(result), mean(pvalues)) if return_pvalues else mean(result)

    if distance_dependent:
        tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
        sqdist_X, sqdist_X_ = tmap(lambda M: cdist(M, M, metric='sqeuclidean'), [X, X_])
        D = remove_diagonal(sqdist_X)
        D_ = remove_diagonal(sqdist_X_)
        scores_X, scores_X_ = (-D, -D_) if isweightedtau else (D, D_)
        for i in range(len(X)):
            corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
            result.append(round(corr, 12))
            pvalues.append(round(pvalue, 12))
    else:  # pragma: no cover
        raise Exception(f"Not implemented yet; it is an open problem")
        #     for i in range(len(X)):
    #         corr, pvalue = sortedness(X, X_, i, f=f, distance_dependent=False, return_pvalues=True,
    #                                   parallel=parallel, parallel_n_trigger=parallel_n_trigger, parallel_kwargs=parallel_kwargs, **kwargs)
    #         result.append(round(corr, 12))
    #         pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=float)
    if return_pvalues:
        return np.array(list(zip(result, pvalues)))
    return result


def pwsortedness(X, X_, i=None, symmetric=True, parallel=True, parallel_n_trigger=200, batches=10, debug=False, dist=None, cython=False, **parallel_kwargs):
    """
    Local pairwise sortedness (Λ𝜏w) based on Sebastiano Vigna weighted Kendall's 𝜏

    Importance rankings are calculated internally based on proximity of each pair to the point of interest.

    # TODO?: add flag to break extremely rare cases of ties that persist after projection (implies a much slower algorithm)
        This probably doesn't make any difference on the res, except on categorical, pathological or toy datasets
        Values can be lower due to the presence of ties, but only when the projection isn't prefect for all points.
        In the end, it might be even desired to penalize ties, as they don't exactly contribute to a stronger ordering and are (probabilistically) easier to be kept than a specific order.

    Parameters
    ----------
    X
        Original dataset
    X_
        Projected points
    i
        None:   calculate pwsortedness for all instances
        `int`:  index of the instance of interest
    symmetric
        True: Take the mean between extrusion and intrusion emphasis
            Equivalent to `(pwsortedness(a, b) + pwsortedness(b, a)) / 2` at a slightly lower cost.
            Might increase memory usage.
        False: Weight by original distances (extrusion emphasis), not the projected distances.
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
        X and X_ should be None
    cython
        Whether to:
            (True) improve speed by ~2x; or,
            (False) be more compatible/portable.


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
    (0.649315577592, 0.753429143832, 0.834601601062)
    >>> r = pwsortedness(original, projected2[:, 1:])
    >>> min(r), round(mean(r), 12), max(r)
    (0.035312055682, 0.2002329034, 0.352491282966)
    >>> r = pwsortedness(original, projectedrnd)
    >>> min(r), round(mean(r), 12), max(r)
    (-0.168611098044, -0.079882538998, 0.14442446342)
    >>> pwsortedness(original, projected1)[1]
    0.649315577592
    >>> pwsortedness(original, projected1, cython=True)[1]
    0.649315577592
    >>> pwsortedness(original, projected1, i=1)
    0.649315577592
    >>> pwsortedness(original, projected1, symmetric=False, cython=True)[1]
    0.730078995423
    >>> pwsortedness(original, projected1, symmetric=False, i=1)
    0.730078995423
    >>> pwsortedness(original, projected1, symmetric=False)
    array([0.75892647, 0.730079  , 0.83496865, 0.73161226, 0.75376525,
           0.83301104, 0.76695755, 0.74759156, 0.81434161, 0.74067221,
           0.74425225, 0.83731035])
    """
    npoints = len(X) if X is not None else len(dist[0])  # pragma: no cover
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and npoints > parallel_n_trigger else map
    if debug:  # pragma: no cover
        print(1)
    thread = lambda M: -pdist(M, metric="sqeuclidean")
    scores_X, scores_X_ = tmap(thread, [X, X_]) if X is not None else (-squareform(dist[0]), -squareform(dist[1]))
    if debug:  # pragma: no cover
        print(2)

    def makeM(E):
        n = len(E)
        m = (n ** 2 - n) // 2
        M = np.zeros((m, E.shape[1]))
        c = 0
        for i in range(n - 1):  # a bit slow, but only a fraction of wtau (~5%)
            h = n - i - 1
            d = c + h
            M[c:d] = E[i] + E[i + 1:]
            c = d
        del E
        gc.collect()
        return M.T

    if symmetric:
        D, D_ = tmap(squareform, [-scores_X, -scores_X_]) if dist is None else (dist[0], dist[1])
    else:
        D = squareform(-scores_X) if dist is None else dist[0]

    if i is None:
        n = len(D)
        if symmetric:
            M, M_ = pmap(makeM, [D, D_])
            R_ = rank_alongrow(M_, step=n // batches, parallel=parallel, **parallel_kwargs).T
            del M_
        else:
            M = makeM(D)
        R = rank_alongrow(M, step=n // batches, parallel=parallel, **parallel_kwargs).T
        del M
        gc.collect()
        if cython:
            from sortedness.wtau import parwtau
            res = parwtau(scores_X, scores_X_, npoints, R, parallel=parallel, **parallel_kwargs)
            del R
            if not symmetric:
                gc.collect()
                return np.round(res, 12)

            res_ = parwtau(scores_X, scores_X_, npoints, R_, parallel=parallel, **parallel_kwargs)
            del R_
            gc.collect()
            return np.round((res + res_) / 2, 12)
        else:
            def thread(r):
                corr = weightedtau(scores_X, scores_X_, rank=r)[0]
                return round(corr, 12)

            gen = (R[:, i] for i in range(len(X)))
            res = np.array(list(pmap(thread, gen)), dtype=float)
            del R
            if not symmetric:
                gc.collect()
                return res

            gen = (R_[:, i] for i in range(len(X_)))
            res_ = np.array(list(pmap(thread, gen)), dtype=float)
            del R_
            gc.collect()
            return np.round((res + res_) / 2, 12)

    if symmetric:
        M, M_ = pmap(makeM, [D[:, i:i + 1], D_[:, i:i + 1]])
        thread = lambda M: rankdata(M, axis=1, method="average")
        r, r_ = [r[0].astype(int) - 1 for r in tmap(thread, [M, M_])]
        s1 = weightedtau(scores_X, scores_X_, r)[0]
        s2 = weightedtau(scores_X, scores_X_, r_)[0]
        return round((s1 + s2) / 2, 12)

    M = makeM(D[:, i:i + 1])
    r = rankdata(M, axis=1, method="average")[0].astype(int) - 1
    return round(weightedtau(scores_X, scores_X_, r)[0], 12)


def rsortedness(X, X_, i=None, symmetric=True, f=weightedtau, return_pvalues=False, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):  # pragma: no cover
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
    X
        Original dataset
    X_
        Projected points
    i
        None:   calculate rsortedness for all instances
        `int`:  index of the instance of interest
    symmetric
        True: Take the mean between extrusion and intrusion emphasis
            Equivalent to `(rsortedness(a, b) + rsortedness(b, a)) / 2` at a slightly lower cost.
            Might increase memory usage.
        False: Weight by original distances (extrusion emphasis), not the projected distances.
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


    # >>> ll = [[i, ] for i in range(17)]
    # >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    # >>> b.ravel()
    # array([ 0, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    # >>> #r = rsortedness(a, b)
    # >>> #min(r), max(r)
    # (-0.707870893072, 0.962964134515)
    #
    # >>> rnd = np.random.default_rng(0)
    # >>> rnd.shuffle(ll)
    # >>> b = np.array(ll)
    # >>> b.ravel()
    # array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    # >>> r = rsortedness(b, a)
    # >>> r
    # array([ 0.36861667, -0.07147685,  0.39350142, -0.04581926, -0.03951645,
    #         0.31100414, -0.18107755,  0.28268222, -0.29248869,  0.19177107,
    #         0.48076521, -0.17640674,  0.13098522,  0.34833996,  0.01844146,
    #        -0.58291518,  0.34742337])
    # >>> min(r), max(r), round(mean(r), 12)
    # (-0.582915181328, 0.480765206133, 0.087284118913)
    # >>> rsortedness(b, a, f=kendalltau, return_pvalues=True)
    # array([[ 0.2316945 ,  0.04863508],
    #        [-0.37005403,  0.21347624],
    #        [ 0.17618709,  0.04863508],
    #        [-0.35418588,  0.21347624]])
    """
    isweightedtau = False
    if hasattr(f, "isweightedtau") and f.isweightedtau:
        isweightedtau = True
        if not symmetric:
            if "rank" in kwargs:
                raise Exception(f"Cannot set `symmetric=False` and provide `rank` at the same time.")
            kwargs["rank"] = None
    elif not symmetric:
        raise Exception(f"`symmetric=False` not implemented for custom `f`")
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
    try:
        from shelchemy.lazy import ichunks
    except Exception as e:
        print("please install shelchemy library.")
        exit()
    if i is None:
        jobs = pmap(thread, ichunks(range(npoints), 15, asgenerators=False))
        for corrs, pvalues in jobs:
            result.extend(corrs)
            pvalues.extend(pvalues)

        result = np.array(result, dtype=float)
        if return_pvalues:
            return np.array(list(zip(result, pvalues)))
        return result

    corr, pvalue = f(scores_X[i], scores_X_[i], **kwargs)
    if return_pvalues:
        return round(corr, 12), pvalue
    return round(corr, 12)


def stress(X, X_, i=None, metric=True, parallel=True, parallel_size_trigger=10000, **parallel_kwargs):
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
    >>> original
    array([[ 1.12573022, 11.86789514],
           [ 1.64042265, 12.10490012],
           [ 0.46433063, 12.36159505],
           [ 2.30400005, 12.94708096],
           [ 0.29626476, 10.73457853],
           [ 0.37672554, 12.04132598],
           [-1.32503077, 11.78120834],
           [-0.24591095, 11.26773265],
           [ 0.45574102, 11.68369984],
           [ 1.41163054, 13.04251337],
           [ 0.87146534, 13.36646347],
           [ 0.33480533, 12.35151007]])
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
    (0.073383317103, 0.440609121637, array([0.26748441, 0.31603101, 0.24636389, 0.07338332, 0.34571508,
           0.19548442, 0.1800883 , 0.16544039, 0.2282494 , 0.16405274,
           0.44060912, 0.27058614]))
    >>> stress(original, projected)
    array([0.26748441, 0.31603101, 0.24636389, 0.07338332, 0.34571508,
           0.19548442, 0.1800883 , 0.16544039, 0.2282494 , 0.16405274,
           0.44060912, 0.27058614])
    >>> stress(original, projected, metric=False)
    array([0.36599664, 0.39465927, 0.27349092, 0.25096851, 0.31476019,
           0.27612935, 0.3064739 , 0.26141414, 0.2635681 , 0.25811772,
           0.36113025, 0.29740821])
    >>> stress(original, projected, 1)
    0.316031007598
    >>> stress(original, projected, 1, metric=False)
    0.39465927169

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
        thread = (lambda M, m: cdist(M, M, metric=m)) if i is None else (lambda M, m: cdist(M[i:i + 1], M, metric=m))
        D, Dsq_ = tmap(thread, [X, X_], ["Euclidean", "sqeuclidean"])
        Dsq_ /= Dsq_.max(axis=1, keepdims=True)
        D_ = sqrt(Dsq_)
    else:
        thread = (lambda M, m: rankdata(cdist(M, M, metric=m), method="average", axis=1) - 1) if i is None else (lambda M, m: rankdata(cdist(M[i:i + 1], M, metric=m), method="average", axis=1) - 1)
        D, Dsq_ = tmap(thread, [X, X_], ["Euclidean", "sqeuclidean"])
        Dsq_ /= Dsq_.max(axis=1, keepdims=True)
        D_ = sqrt(Dsq_)

    D /= D.max(axis=1, keepdims=True)
    s = ((D - D_) ** 2).sum(axis=1) / 2
    result = np.round(np.sqrt(s / (Dsq_.sum(axis=1) / 2)), 12)
    return result if i is None else result[0]
