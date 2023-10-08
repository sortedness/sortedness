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

from functools import partial
from itertools import islice
from math import ceil, log, sqrt

import numpy as np
import pathos.multiprocessing as mp
from numpy import eye, mean, ndarray, array
from numpy.random import permutation
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, weightedtau
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors

from sortedness.local import common


# teste de history

def sortedness(X, X_, i=None, symmetric=True, f=kendalltau, return_pvalues=False, seed=10, minsample=4, memory_MB=2000, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    """
     Calculate sortedness probabilistically, i.e., sampling neighbors instead of weighting

     Functions available as scipy correlation coefficients:
         ρ-sortedness (Spearman),
         𝜏-sortedness (Kendall's 𝜏),

    Note:
        Categorical, or pathological data might present values lower than one due to the presence of ties even with a perfect projection.
        Depending on the chosen correlation coefficient, ties are penalized, as they do not contribute to establishing any order.

    Hint:
        Swap two points A and B at X_ to be able to calculate sortedness between A and B in the same space (i.e., originally, `X = X_`):
            `X = [A, B, C, ..., Z]`
            `X_ = [B, A, C, ..., Z]`
            `sortedness(X, X_, i=0)`

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
            Same as `(sortedness(a, b, symmetric=False) + sortedness(b, a, symmetric=False)) / 2`
        False: Weight by original distances (extrusion emphasis), not the projected distances.
    f
        Agreement function:
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
    seed
        Seed for pseudorandom sampling of neighbors
    minsample
        Minimum length allowed after sampling to feed the correlation function
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
         ndarray containing a sortedness value per row, or a single float (include pvalues as a second value/column if requested)

    >>> ll = [[i] for i in range(37)]
    >>> a, b = np.array(ll), np.array(ll[0:1] + list(reversed(ll[1:])))
    >>> b.ravel()
    array([ 0, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
           20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,
            3,  2,  1])
    >>> r = np.average(np.array([sortedness(a, b, seed=i) for i in range(1000)]), axis=0)
    >>> from sortedness import sortedness as s
    >>> np.round(r - s(a, b),2)
    >>> min(r), max(r), r
    (-1.0, 0.999576181006, array([-1.        ,  0.80423925,  0.89071337,  0.92359576,  0.94227454,
            0.95491184,  0.96434749,  0.97185032,  0.97808063,  0.98341964,
            0.98810467,  0.99229223,  0.99609031,  0.99957618,  0.99957618,
            0.99609031,  0.99229223,  0.98810467,  0.98341964,  0.97808063,
            0.97185032,  0.96434749,  0.95491184,  0.94227454,  0.92359576,
            0.89071337,  0.80423925]))

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

    # Measure sortedness between two points in the same space.
    >>> M = original.copy()
    >>> M[0], M[1] = original[1], original[0]
    >>> sortedness(M, original, 0)
    0.547929184934

    >>> s = sortedness(original, projected2)
    >>> min(s), max(s), s
    (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    >>> s = sortedness(original, projected1)
    >>> min(s), max(s)
    (0.393463224666, 0.944810120534)
    >>> s = sortedness(original, projectedrnd)
    >>> min(s), max(s)
    (-0.648305479567, 0.397019507592)

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
    >>> sortedness(projected, original, 1, symmetric=False)
    0.616485374665
    >>> sortedness(original, projected)[1]
    0.519562134793
    >>> sortedness([[1,2,3,3],[1,2,7,3],[3,4,7,8],[5,2,6,3],[3,5,4,8],[2,7,7,5]], [[7,1,2,3],[3,7,7,3],[5,4,5,6],[9,7,6,3],[2,3,5,1],[1,2,6,3]], 1)
    -1.0


    >>> ([sortedness(a, b, 0, seed=i) for i in range(100)])
    -0.333333333333
    """

    # >>> r = sortedness(a, b)
    # >>> min(r), max(r)
    # (-1.0, 0.998638259786)
    #
    # # Measure sortedness between two points in the same space.
    # >>> M = original.copy()
    # >>> M[0], M[1] = original[1], original[0]
    # >>> sortedness(M, original, 0)
    # 0.547929184934
    #
    # >>> rnd = np.random.default_rng(0)
    # >>> rnd.shuffle(ll)
    # >>> b = np.array(ll)
    # >>> b.ravel()
    # array([ 2, 10,  3, 11,  0,  4,  7,  5, 16, 12, 13,  6,  9, 14,  8,  1, 15])
    # >>> r = sortedness(a, b)
    # >>> r
    # array([ 0.24691868, -0.17456491,  0.19184376, -0.18193532,  0.07175694,
    #         0.27992254,  0.04121859,  0.16249574, -0.03506842,  0.27856259,
    #         0.40866965, -0.07617887,  0.12184064,  0.24762942, -0.05049511,
    #        -0.46277399,  0.12193493])
    # >>> min(r), max(r)
    # (-0.462773990559, 0.408669653064)
    # >>> round(mean(r), 12)
    # 0.070104521222
    #
    # >>> import numpy as np
    # >>> from functools import partial
    # >>> from scipy.stats import spearmanr, weightedtau
    # >>> me = (1, 2)
    # >>> cov = eye(2)
    # >>> rng = np.random.default_rng(seed=0)
    # >>> original = rng.multivariate_normal(me, cov, size=12)
    # >>> from sklearn.decomposition import PCA
    # >>> projected2 = PCA(n_components=2).fit_transform(original)
    # >>> projected1 = PCA(n_components=1).fit_transform(original)
    # >>> np.random.seed(0)
    # >>> projectedrnd = permutation(original)
    #
    # >>> s = sortedness(original, original)
    # >>> min(s), max(s), s
    # (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # >>> s = sortedness(original, projected2)
    # >>> min(s), max(s), s
    # (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    # >>> s = sortedness(original, projected1)
    # >>> min(s), max(s)
    # (0.393463224666, 0.944810120534)
    # >>> s = sortedness(original, projectedrnd)
    # >>> min(s), max(s)
    # (-0.648305479567, 0.397019507592)
    #
    # >>> sortedness(original, original, f=kendalltau, return_pvalues=True)
    # array([[1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08],
    #        [1.0000e+00, 5.0104e-08]])
    # >>> sortedness(original, projected2, f=kendalltau)
    # array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    # >>> sortedness(original, projected1, f=kendalltau)
    # array([0.56363636, 0.52727273, 0.81818182, 0.96363636, 0.70909091,
    #        0.85454545, 0.74545455, 0.92727273, 0.85454545, 0.89090909,
    #        0.6       , 0.74545455])
    # >>> sortedness(original, projectedrnd, f=kendalltau)
    # array([ 0.2       , -0.38181818,  0.23636364, -0.09090909, -0.05454545,
    #         0.23636364, -0.09090909,  0.23636364, -0.63636364, -0.01818182,
    #        -0.2       , -0.01818182])
    #
    # >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    # >>> sortedness(original, original, f=wf, return_pvalues=True)
    # array([[ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan],
    #        [ 1., nan]])
    # >>> sortedness(original, projected2, f=wf)
    # array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    # >>> sortedness(original, projected1, f=wf)
    # array([0.89469168, 0.89269637, 0.92922928, 0.99721669, 0.86529591,
    #        0.97806422, 0.94330979, 0.99357377, 0.87959707, 0.92182767,
    #        0.87256459, 0.87747329])
    # >>> sortedness(original, projectedrnd, f=wf)
    # array([ 0.23771513, -0.2790059 ,  0.3718005 , -0.16623167,  0.06179047,
    #         0.40434396, -0.00130294,  0.46569739, -0.67581876, -0.23852189,
    #        -0.39125007,  0.12131153])
    # >>> np.random.seed(14980)
    # >>> projectedrnd = permutation(original)
    # >>> sortedness(original, projectedrnd)
    # array([ 0.24432153, -0.19634576, -0.00238081, -0.4999116 , -0.01625951,
    #         0.22478766,  0.07176118, -0.48092843,  0.19345964, -0.44895295,
    #        -0.42044773,  0.06942218])
    # >>> sortedness(original, np.flipud(original))
    # array([-0.28741742,  0.36769361,  0.06926091,  0.02550202,  0.21424544,
    #        -0.3244699 , -0.3244699 ,  0.21424544,  0.02550202,  0.06926091,
    #         0.36769361, -0.28741742])
    # >>> original = np.array([[0],[1],[2],[3],[4],[5],[6]])
    # >>> projected = np.array([[6],[5],[4],[3],[2],[1],[0]])
    # >>> sortedness(original, projected)
    # array([1., 1., 1., 1., 1., 1., 1.])
    # >>> projected = np.array([[0],[6],[5],[4],[3],[2],[1]])
    # >>> sortedness(original, projected)
    # array([-1.        ,  0.51956213,  0.81695345,  0.98180162,  0.98180162,
    #         0.81695345,  0.51956213])
    # >>> sortedness(original, projected, 1)
    # 0.519562134793
    # >>> sortedness([[1,2,3,3],[1,2,7,3],[3,4,7,8],[5,2,6,3],[3,5,4,8],[2,7,7,5]], [[7,1,2,3],[3,7,7,3],[5,4,5,6],[9,7,6,3],[2,3,5,1],[1,2,6,3]], 1)
    # -1.0
    # """
    if parallel_kwargs is None:
        parallel_kwargs = {}
    rnd = np.random.default_rng(seed)
    n = len(X) - 1

    # harmonic weights as probabilities
    if "weigher" in kwargs:
        w = kwargs.pop("weigher")
    else:
        w = (lambda r: 1 / (r + 1))
    p = array([w(r) for r in range(n)])  # todo: adopt linear extra penalty?
    # p = p / sum(p)
    w = int(ceil(log(n) + 0.577))  # total weight of harmonic sequence
    print(f"{w=}")

    if i is None:
        gen = pairwise_distances_chunked(X, n_jobs=-1, working_memory=memory_MB)
        gen_ = pairwise_distances_chunked(X_, n_jobs=-1, working_memory=memory_MB)
        i, res = 0, []
        m = 0
        for chunk, chunk_ in zip(gen, gen_):
            for d, d_ in zip(chunk, chunk_):
                d, d_ = np.delete(d, i), np.delete(d_, i)

                for j in range(1):
                    # selected_ranks = rnd.choice(n, size=w, replace=False, p=p, shuffle=False)
                    while len((selected_ranks := np.flatnonzero(rnd.uniform(0, 1, n) <= p))) < minsample:
                        pass
                    idxs = np.argpartition(d, selected_ranks)[selected_ranks]
                    idxs_ = np.argpartition(d_, selected_ranks)[selected_ranks]
                    s = f(d[idxs], d_[idxs], **kwargs)[0]
                    s_ = f(d_[idxs_], d[idxs_], **kwargs)[0]
                    r = (s + s_) / 2
                    res.append(r)

                i += 1
                if i == 1000:
                    return array(res, dtype=float)
        return array(res, dtype=float)
        # todo: symmetric, pvalues
    else:
        pmap = None
        if not isinstance(X, ndarray):
            X, X_ = array(X), array(X_)
        x, x_ = X[i], X_[i]
        X = np.delete(X, i, axis=0)
        X_ = np.delete(X_, i, axis=0)
        d_ = np.sum((X_ - x_) ** 2, axis=1)
        d = np.sum((X - x) ** 2, axis=1)

        while len((selected_ranks := np.flatnonzero(rnd.uniform(0, 1, n) < p))) < minsample:
            pass
        idxs = np.argpartition(d, selected_ranks)[selected_ranks]
        scores_X = d[idxs]
        scores_X_ = d_[idxs]
        # print(len(idxs), idxs, kendalltau(scores_X,scores_X_))

    return common(scores_X, scores_X_, i, symmetric, f, False, return_pvalues, pmap, kwargs)


def quad(X, X_, i=None, symmetric=True, f=kendalltau, return_pvalues=False, seed=10, minsample=4, memory_MB=2000, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    if parallel_kwargs is None:
        parallel_kwargs = {}
    rnd = np.random.default_rng(seed)
    n = len(X)

    # harmonic weights as probabilities
    p = [1 / r for r in range(1, n)]  # todo: adopt linear extra penalty?
    # w = 3+int(round(log(n) + 0.577))  # total weight  # todo: see above, this will change things here
    if not isinstance(X, ndarray):
        X, X_ = array(X), array(X_)

    if i is None:
        M, M_ = X, X_
        ds, ds_ = [], []
        old = 0
        while len(M) != old and len(M) > 2:
            old = len(M)
            kdtree = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(M)
            d, idx = (ret := kdtree.kneighbors(X))[0][:, 1], ret[1][:, 1]
            d_ = M_[idx, 0]
            ds.append(d)
            ds_.append(d_)
            M = np.delete(M, idx, axis=0)
            M_ = np.delete(M_, idx, axis=0)
        scores = np.column_stack(ds)
        scores_ = np.column_stack(ds_)
        tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and n > parallel_n_trigger else map
        pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and n > parallel_n_trigger else map
        if return_pvalues:
            res = list(tmap(f, scores, scores_))
        else:
            res = list(tup[0] for tup in pmap(f, scores, scores_))
        return array(res, dtype=float)
    else:
        pmap = None
        if not isinstance(X, ndarray):
            X, X_ = array(X), array(X_)
        x, x_ = X[i], X_[i]
        X = np.delete(X, i, axis=0)
        X_ = np.delete(X_, i, axis=0)
        d_ = np.sum((X_ - x_) ** 2, axis=1)
        d = np.sum((X - x) ** 2, axis=1)

        while len((selected_ranks := np.flatnonzero(rnd.uniform(0, 1, n) < p))) < minsample:
            pass
        idxs = np.argpartition(d, selected_ranks)[selected_ranks]
        scores_X = d[idxs]
        scores_X_ = d_[idxs]
        # print(len(idxs), idxs, kendalltau(scores_X,scores_X_))

    return common(scores_X, scores_X_, i, symmetric, f, False, return_pvalues, pmap, kwargs)


def locglo(X, X_, i=None, symmetric=True, f=kendalltau, return_pvalues=False, seed=10, minsample=4, sample_size=1000, memory_MB=2000, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    if parallel_kwargs is None:
        parallel_kwargs = {}
    n = len(X) - 1

    rnd = np.random.default_rng(seed)
    perm = rnd.permutation(n + 1)
    if not isinstance(X, ndarray):
        X, X_ = array(X), array(X_)
    # print(X.shape, sorted(perm))
    X = X[perm]
    # print(3333333333333333333333333333333333333333)
    X_ = X_[perm]
    L = min(sample_size, int(sqrt(n + 1)))
    gidxs = rnd.permutation(n + 1)

    if i is None:
        gen = pairwise_distances_chunked(X, n_jobs=-1, working_memory=memory_MB)
        gen_ = pairwise_distances_chunked(X_, n_jobs=-1, working_memory=memory_MB)
        i, res = 0, []
        m = 0
        for chunk, chunk_ in zip(gen, gen_):
            for d, d_ in zip(chunk, chunk_):
                d0, d0_ = d, d_
                d, d_ = np.delete(d, i), np.delete(d_, i)

                selected_ranks = array(list(range(10)))
                idxs = np.argpartition(d, selected_ranks)[:10]
                idxs_ = np.argpartition(d_, selected_ranks)[:10]
                s = f(d[idxs], d_[idxs], **kwargs)[0]
                s_ = f(d_[idxs_], d[idxs_], **kwargs)[0]
                r1 = (s + s_) / 2

                take = gidxs[:L]  # this will rarely select x as a neigbor of itself by accident
                print(take[:4])
                break
                r2 = kendalltau(d0[take], d0_[take], **kwargs)[0]  # f should be symmetric
                gidxs = gidxs[L:]

                r = (r1 + r2) / 2
                res.append(r)

                i += 1
                if i >= L:
                    print(len(res))
                    return array(res, dtype=float)


def locgloq(X, X_, i=None, symmetric=True, f=kendalltau, return_pvalues=False, seed=10, minsample=4, sample_size=1000, memory_MB=2000, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    if parallel_kwargs is None:
        parallel_kwargs = {}
    n = len(X) - 1

    rnd = np.random.default_rng(seed)
    perm = rnd.permutation(n + 1)
    # print(X.shape, sorted(perm))
    if not isinstance(X, ndarray):
        X, X_ = array(X), array(X_)
    X = X[perm]
    # print(3333333333333333333333333333333333333333)
    X_ = X_[perm]
    nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(X)
    nbrs_ = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(X_)
    L = min(sample_size, int(sqrt(n + 2)))
    gidxs = rnd.permutation(n + 1)

    if i is None:
        res = []
        m = 0
        for i in range(L):  # todo: remove unneeded permutation above (slow for big datasets), as we can just sample enough i values
            x = X[i:i + 1]
            x_ = X_[i:i + 1]

            d, idx = nbrs.kneighbors(x, 11)
            d_ = ((X_[idx[0, 1:]] - x_) ** 2).sum(axis=1)
            S = f(d[0, 1:], d_, **kwargs)[0]

            d_, idx_ = nbrs_.kneighbors(x_, 11)
            d = ((X[idx_[0, 1:]] - x) ** 2).sum(axis=1)
            s_ = f(d_[0, 1:], d, **kwargs)[0]

            r1 = (S + s_) / 2

            ta = gidxs[:L]
            d0 = cdist(x, X[ta], metric="sqeuclidean")
            d0_ = cdist(x_, X_[ta], metric="sqeuclidean")
            r2 = kendalltau(d0, d0_, **kwargs)[0]  # f should be symmetric
            gidxs = gidxs[L:]

            r = (r1 + r2) / 2
            res.append(r)

            i += 1
        return array(res, dtype=float)


def locgloth(X, X_, i=None, symmetric=True, f=kendalltau, return_pvalues=False, seed=10, minsample=4, sample_size=1000, memory_MB=2000, parallel=True, parallel_n_trigger=500, parallel_kwargs=None, **kwargs):
    if parallel_kwargs is None:
        parallel_kwargs = {}
    n = len(X) - 1

    rnd = np.random.default_rng(seed)
    perm = rnd.permutation(n + 1)
    if not isinstance(X, ndarray):
        X, X_ = array(X), array(X_)
    # print(X.shape, sorted(perm))
    X = X[perm]
    # print(3333333333333333333333333333333333333333)
    X_ = X_[perm]
    L = min(sample_size, int(sqrt(n + 1)))
    gidxs = rnd.permutation(n + 1)

    if i is None:
        gen = pairwise_distances_chunked(X, n_jobs=-1, working_memory=memory_MB)
        gen_ = pairwise_distances_chunked(X_, n_jobs=-1, working_memory=memory_MB)
        i, res = 0, []
        m = 0
        tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel and n > parallel_n_trigger else map
        pmap = mp.ProcessingPool(**parallel_kwargs).imap if parallel and n > parallel_n_trigger else map

        def th(d, d_, take):
            d0, d0_ = d, d_
            d, d_ = np.delete(d, i), np.delete(d_, i)

            selected_ranks = array(list(range(10)))
            idxs = np.argpartition(d, selected_ranks)[:10]
            idxs_ = np.argpartition(d_, selected_ranks)[:10]
            s = f(d[idxs], d_[idxs], **kwargs)[0]
            s_ = f(d_[idxs_], d[idxs_], **kwargs)[0]
            r1 = (s + s_) / 2

            print(take[:4])
            exit()
            r2 = kendalltau(d0[take], d0_[take], **kwargs)[0]  # f should be symmetric
            r = (r1 + r2) / 2
            return r

        jobs = tmap
        sp = np.split(gidxs, L)
        for chunk, chunk_ in islice(zip(gen, gen_), L):
            res.extend(jobs(th, chunk, chunk_, sp[:len(chunk)]))
            sp = sp[len(chunk):]

        print(len(res))
        return array(res, dtype=float)
