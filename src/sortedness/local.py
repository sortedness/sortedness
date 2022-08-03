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

from functools import partial

import numpy as np
from numpy import eye, argsort, nan
from numpy.linalg import norm
from numpy.random import shuffle, permutation
from scipy.stats import spearmanr, weightedtau, kendalltau, rankdata
from sklearn.decomposition import PCA

from sortedness.rank import (
    rank_by_distances,
    rdist_by_index_lw,
    rdist_by_index_iw,
    euclidean__n_vs_1,
    differences__n_vs_1,
)


def ushaped_decay_f(n):
    def f(i):
        x = (n - i) / n
        return 4 * x**2 - 4 * x + 1

    return f


# noinspection PyTypeChecker
def sortedness(X, X_, f=spearmanr, return_pvalues=False, weigher=None, normalized=True):
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
     (0.734265734266, 0.993006993007, array([0.78321678, 0.73426573, 0.94405594, 0.99300699, 0.86713287,
            0.95804196, 0.9020979 , 0.97902098, 0.96503497, 0.97902098,
            0.7972028 , 0.88811189]))
     >>> s = sortedness(original, projectedrnd)
     >>> min(s), max(s), s
     (-0.398601398601, 0.496503496503, array([ 0.3986014 , -0.16783217,  0.46153846,  0.1048951 ,  0.18881119,
             0.4965035 ,  0.12587413,  0.43356643, -0.3986014 ,  0.16783217,
             0.03496503,  0.12587413]))

     >>> from sortedness.kruskal import kruskal
     >>> s = kruskal(original, original, f=partial(rank_by_distances))
     >>> min(s), max(s), s
     (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
     >>> s = kruskal(original, projected2, f=partial(rank_by_distances))
     >>> min(s), max(s), s
     (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
     >>> s = kruskal(original, projected1, f=partial(rank_by_distances))
     >>> min(s), max(s), s
     (0.062869461346, 0.387553387882, array([0.35004235, 0.38755339, 0.17782169, 0.06286946, 0.27404163,
            0.1539981 , 0.23523598, 0.1088931 , 0.14058039, 0.1088931 ,
            0.33856241, 0.25147785]))
    >>> s = kruskal(original, projectedrnd, f=partial(rank_by_distances))
     >>> min(s), max(s), s
     (0.533465069369, 0.889108448949, array([0.5830274 , 0.81245249, 0.55167728, 0.71128676, 0.67712482,
            0.53346507, 0.70290195, 0.56582515, 0.88910845, 0.68582485,
            0.73854895, 0.70290195]))

     >>> s, pvalues = sortedness(original, original, f=kendalltau, return_pvalues=True)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> pvalues
     [4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09]
     >>> s = sortedness(original, projected2, f=kendalltau)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> s = sortedness(original, projected1, f=kendalltau)
     >>> min(s), max(s), s
     (0.606060606061, 0.969696969697, array([0.63636364, 0.60606061, 0.84848485, 0.96969697, 0.75757576,
            0.87878788, 0.78787879, 0.93939394, 0.87878788, 0.90909091,
            0.66666667, 0.78787879]))
    >>> s = sortedness(original, projectedrnd, f=kendalltau)
     >>> min(s), max(s), s
     (-0.363636363636, 0.363636363636, array([ 0.33333333, -0.15151515,  0.36363636,  0.09090909,  0.12121212,
             0.36363636,  0.09090909,  0.36363636, -0.36363636,  0.15151515,
             0.        ,  0.15151515]))

     >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
     >>> s, pvalues = sortedness(original, original, f=wf, return_pvalues=True)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> pvalues
     [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
     >>> s = sortedness(original, projected2, f=wf)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> s = sortedness(original, projected1, f=wf)
     >>> min(s), max(s), s
     (0.878046135266, 0.99748013867, array([0.9046595 , 0.90285305, 0.93592798, 0.99748014, 0.87804614,
            0.98014052, 0.94867572, 0.99418203, 0.89099364, 0.92922697,
            0.88462681, 0.88907089]))
    >>> s = sortedness(original, projectedrnd, f=wf)
     >>> min(s), max(s), s
     (-0.517196452192, 0.516271063981, array([ 0.30986815, -0.15794336,  0.43126186, -0.05584362,  0.15059539,
             0.46072496,  0.093474  ,  0.51627106, -0.51719645, -0.12129132,
            -0.25956322,  0.20448257]))

     >>> wf = partial(weightedtau, weigher=ushaped_decay_f(n=len(original)))
     >>> s, pvalues = sortedness(original, original, f=wf, return_pvalues=True)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> pvalues
     [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
     >>> s = sortedness(original, projected2, f=wf)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> s = sortedness(original, projected1, f=wf)
     >>> min(s), max(s), s
     (0.795765877958, 0.983810709838, array([0.80821918, 0.79576588, 0.93524284, 0.98381071, 0.88542964,
            0.94271482, 0.91656289, 0.97633873, 0.89912827, 0.93150685,
            0.84059776, 0.89290162]))
     >>> s = sortedness(original, projectedrnd, f=wf)
     >>> min(s), max(s), s
     (-0.252801992528, 0.572851805729, array([ 0.39726027, -0.03611457,  0.48069738,  0.15566625,  0.24408468,
             0.57285181,  0.16562889,  0.49937733, -0.25280199,  0.14445828,
            -0.05230386,  0.25653798]))

     >>> s = sortedness(original, original, f=None)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> pvalues
     [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
     >>> s = sortedness(original, projected2, f=None)
     >>> min(s), max(s), s
     (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
     >>> s = sortedness(original, projected1, f=None)
     >>> min(s), max(s), s
     (0.3845013898860591, 0.914950019329497, array([0.38450139, 0.48271387, 0.7969938 , 0.91495002, 0.74879881,
            0.70840007, 0.66547008, 0.85116253, 0.81856004, 0.83840504,
            0.41072513, 0.79243755]))
     >>> s = sortedness(original, projectedrnd, f=None)
     >>> min(s), max(s), s
     (-0.37488264207211563, 0.26744721194375687, array([ 0.12002725, -0.26401392,  0.22603597, -0.08439646, -0.04187147,
             0.26744721, -0.09755896,  0.08033725, -0.37488264,  0.00915852,
             0.02333352, -0.07022146]))

     >>> s = sortedness(original, original, f=None, normalized=False)
     >>> min(s), max(s), s
     (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
     >>> s = sortedness(original, projected2, f=None, normalized=False)
     >>> min(s), max(s), s
     (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
     >>> s = sortedness(original, projected1, f=None, normalized=False)
     >>> min(s), max(s), s
     (0.666666666667, 4.824603174603, array([4.82460317, 4.0547619 , 1.59126984, 0.66666667, 1.96904762,
            2.28571429, 2.62222222, 1.16666667, 1.42222222, 1.26666667,
            4.61904762, 1.62698413]))
    >>> s = sortedness(original, projectedrnd, f=None, normalized=False)
    >>> min(s), max(s), s
    (5.742135642136, 10.777056277056, array([ 6.8976912 ,  9.90800866,  6.06673882,  8.50007215,  8.16673882,
            5.74213564,  8.60324675,  7.20880231, 10.77705628,  7.76673882,
            7.65562771,  8.38896104]))
    >>> s = sortedness(original, np.flipud(original), f=None, normalized=False)
    >>> min(s), max(s), s
    (4.908802308802, 9.508008658009, array([9.03896104, 4.90880231, 6.76197691, 7.32229437, 5.98896104,
           9.50800866, 9.50800866, 5.98896104, 7.32229437, 6.76197691,
           4.90880231, 9.03896104]))

     Parameters
     ----------
     X
         matrix with an instance by row in a given space (often the original one)
     X_
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
         None        =   special internal sortedness function will be used
     return_pvalues
         For scipy correlation functions, return a tuple '«corr, pvalue»' instead of just 'corr'
         This makes more sense for Kendall's tau. [the weighted version might not have yet a established pvalue calculation method at this moment]
         The null hypothesis is that the projection is random, i.e., sortedness = 0.5.
     normalized
         Only for 'f=None'
     weigher
         Only for 'f=None'

     Returns
     -------
         list of sortedness values (or tuples that also include pvalues)
    """
    result, pvalues = [], []
    if f is None:
        if weigher is None:
            weigher = lambda r: 1 / (1 + r)
        weights = [weigher(i) for i in range(len(X))]
        if normalized:
            woa = np.array(range(len(X)), dtype=np.float).reshape(len(X), 1)
            wob = np.array(list(range(len(X) - 1, -1, -1)), dtype=np.float).reshape(len(X), 1)
            worst = ff(woa, woa[0], wob, woa[0], weights)
        for a, b in zip(X, X_):
            t = ff(X, a, X_, b, weights)
            if normalized:
                if t != 0:
                    t /= worst
                t = 1 - (2 * t)
            result.append(t)
            pvalues.append(nan)
    else:
        if weigher is not None:
            raise Exception("Cannot provide both 'f' and 'weigher'.")
        for a, b in zip(X, X_):
            corr, pvalue = f(euclidean__n_vs_1(X, a), euclidean__n_vs_1(X_, b))
            result.append(round(corr, 12))
            pvalues.append(round(pvalue, 12))

    result = np.array(result, dtype=np.float)
    if return_pvalues:
        return result, pvalues
        # return list(zip(result, pvalues))
    return result


def ff(X, a, X_, b, weights):
    t = 0
    for idxa, idxb in zip(rank_by_distances(X, a), rank_by_distances(X_, b)):
        mn, mx = sorted([int(idxa), int(idxb)])
        t += sum(weights[p] for p in range(mn, mx))
    return round(t, 12)


# Still non-reliable proposal attempts: #####################################


def sortedness_(X, X_, f="lw", normalized=False, decay=None):
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
    X
        matrix with an instance by row in a given space (often the original one)
    X_
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
    for a, b in zip(X, X_):
        ranks_ma = rank_by_distances(X, a)
        mb_ = X_[argsort(ranks_ma)]  # Sort mb by using ma ranks.
        ranks = rank_by_distances(mb_, b)
        d = f(argsort(ranks), normalized=normalized, **kwargs)
        result.append(d)
    return result


def asortedness_(X, X_, f=spearmanr, return_pvalues=False, use_kemeny_young=False):  # pragma: no cover
    """
    Calculate the 𝛼-sortedness (a anti-stress alike correlation-based measure that is independent of distance function)
     value for each point
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
    >>> s = asortedness_(original, original)
    >>> min(s), max(s), s
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> projected = PCA(n_components=1).fit_transform(original)
    >>> s = asortedness_(original, projected)
    >>> min(s), max(s), s
    (0.605955395113, 0.972027972028, [0.757899402288, 0.605955395113, 0.944055944056, 0.965034965035, 0.916083916084, 0.928197570548, 0.909090909091, 0.972027972028, 0.951048951049, 0.958041958042, 0.748251748252, 0.944055944056])
    >>> s, pvalues = asortedness_(original, projected, f=kendalltau, return_pvalues=True)
    >>> min(s), max(s), s
    (0.442760992001, 0.909090909091, [0.553911709407, 0.442760992001, 0.848484848485, 0.878787878788, 0.818181818182, 0.809183881932, 0.818181818182, 0.909090909091, 0.848484848485, 0.848484848485, 0.636363636364, 0.818181818182])
    >>> pvalues
    [0.01312710939157066, 0.04622906055239071, 1.6342325370103148e-05, 5.319397680508792e-06, 4.4129288920955584e-05, 0.00026956263420192036, 4.4129288920955584e-05, 1.4655483405483405e-06, 1.6342325370103148e-05, 1.6342325370103148e-05, 0.003181646992410881, 4.4129288920955584e-05]
    >>> s = asortedness_(original, projected, f=weightedtau)
    >>> min(s), max(s), s
    (0.510825422491, 0.952232594366, [0.55533349733, 0.510825422491, 0.839690308181, 0.769474895665, 0.897976286974, 0.834349193877, 0.90387231025, 0.952232594366, 0.894060752607, 0.757105822997, 0.793982653284, 0.821971590447])
    >>> projected = PCA(n_components=2).fit_transform(original)
    >>> s = asortedness_(original, projected)
    >>> min(s), max(s), s   # doctest: +SKIP
    (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> _, pvalues = asortedness_(original, projected, return_pvalues=True)
    >>> pvalues   # doctest: +SKIP
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> np.random.seed(0)
    >>> projected = permutation(original)
    >>> s = asortedness_(original, projected)
    >>> min(s), max(s), s
    (-0.370629370629, 0.468531468531, [0.214036405276, 0.20665530816, 0.468531468531, 0.104895104895, 0.181818181818, 0.462347469103, 0.028021058734, 0.356643356643, -0.370629370629, 0.20979020979, 0.017543967646, 0.43432641037])

    Parameters
    ----------
    use_kemeny_young
    X
        matrix with an instance by row in a given space (often the original one)
    X_
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
        The null hypothesis is that the projection is random, i.e., asortedness = 0.5.

    Returns
    -------
        list of asortedness values (or tuples that also include pvalues)
    """
    result, pvalues = [], []
    for a, b in zip(X, X_):
        diffs_a = differences__n_vs_1(X, a)
        diffs_b = differences__n_vs_1(X_, b)
        ranks_a = rankdata(diffs_a, axis=0)
        ranks_b = rankdata(diffs_b, axis=0)
        if use_kemeny_young:  # closest ranking
            from ranky import kemeny_young

            evaluations_a = kemeny_young(ranks_a, verbose=False)  # KY default: kendall's tau
            evaluations_b = kemeny_young(ranks_b, verbose=False)
        else:  # Manhattan distance
            evaluations_a = norm(ranks_a, axis=1, keepdims=True)  # TODO: afetado por rotacao; propor outro
            evaluations_b = norm(ranks_b, axis=1, keepdims=True)
        corr, pvalue = f(evaluations_a, evaluations_b)
        result.append(round(corr, 12))
        pvalues.append(pvalue)
    if return_pvalues:
        return result, pvalues
        # return list(zip(result, pvalues))
    return result


def asortedness__(X, X_, f=spearmanr, return_pvalues=False):  # pragma: no cover
    """
    Calculate the 𝛼-sortedness (a anti-stress alike correlation-based measure that is independent of distance function)
     value for each point
    Functions available as scipy correlation coefficients:
        ρ-sortedness (Spearman),
        𝜏-sortedness (Kendall's 𝜏),
        w𝜏-sortedness (Sebastiano Vigna weighted Kendall's 𝜏)


    Parameters
    ----------
    X
        matrix with an instance by row in a given space (often the original one)
    X_
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
        The null hypothesis is that the projection is random, i.e., asortedness = 0.5.

    Returns
    -------
        list of asortedness values (or tuples that also include pvalues)
    """
    #  >>> import numpy as np
    #  >>> from functools import partial
    #  >>> from scipy.stats import spearmanr, weightedtau
    #  >>> mean = (1, 2)
    #  >>> cov = eye(2)
    #  >>> rng = np.random.default_rng(seed=0)
    #  >>> original = rng.multivariate_normal(mean, cov, size=12)
    #  >>> projected2 = PCA(n_components=2).fit_transform(original)
    #  >>> projected1 = PCA(n_components=1).fit_transform(original)
    #  >>> np.random.seed(0)
    #  >>> projectedrnd = permutation(original)
    #
    #  >>> s = asortedness__(original, original)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> s = asortedness__(original, projected2)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> s = asortedness__(original, projected1)
    #  >>> min(s), max(s), s
    #  (0.734265734266, 0.993006993007, array([0.78321678, 0.73426573, 0.94405594, 0.99300699, 0.86713287,
    #         0.95804196, 0.9020979 , 0.97902098, 0.96503497, 0.97902098,
    #         0.7972028 , 0.88811189]))
    #  >>> s = asortedness__(original, projectedrnd)
    #  >>> min(s), max(s), s
    #  (-0.398601398601, 0.496503496503, array([ 0.3986014 , -0.16783217,  0.46153846,  0.1048951 ,  0.18881119,
    #          0.4965035 ,  0.12587413,  0.43356643, -0.3986014 ,  0.16783217,
    #          0.03496503,  0.12587413]))
    #
    #  >>> from sortedness.kruskal import kruskal
    #  >>> s = kruskal(original, original, f=partial(rank_by_distances))
    #  >>> min(s), max(s), s
    #  (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    #  >>> s = kruskal(original, projected2, f=partial(rank_by_distances))
    #  >>> min(s), max(s), s
    #  (0.0, 0.0, array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    #  >>> s = kruskal(original, projected1, f=partial(rank_by_distances))
    #  >>> min(s), max(s), s
    #  (0.062869461346, 0.387553387882, array([0.35004235, 0.38755339, 0.17782169, 0.06286946, 0.27404163,
    #         0.1539981 , 0.23523598, 0.1088931 , 0.14058039, 0.1088931 ,
    #         0.33856241, 0.25147785]))
    # >>> s = kruskal(original, projectedrnd, f=partial(rank_by_distances))
    #  >>> min(s), max(s), s
    #  (0.533465069369, 0.889108448949, array([0.5830274 , 0.81245249, 0.55167728, 0.71128676, 0.67712482,
    #         0.53346507, 0.70290195, 0.56582515, 0.88910845, 0.68582485,
    #         0.73854895, 0.70290195]))
    #
    #  >>> s, pvalues = asortedness__(original, original, f=kendalltau, return_pvalues=True)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> pvalues
    #  [4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09, 4.175e-09]
    #  >>> s = asortedness__(original, projected2, f=kendalltau)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> s = asortedness__(original, projected1, f=kendalltau)
    #  >>> min(s), max(s), s
    #  (0.606060606061, 0.969696969697, array([0.63636364, 0.60606061, 0.84848485, 0.96969697, 0.75757576,
    #         0.87878788, 0.78787879, 0.93939394, 0.87878788, 0.90909091,
    #         0.66666667, 0.78787879]))
    # >>> s = asortedness__(original, projectedrnd, f=kendalltau)
    #  >>> min(s), max(s), s
    #  (-0.363636363636, 0.363636363636, array([ 0.33333333, -0.15151515,  0.36363636,  0.09090909,  0.12121212,
    #          0.36363636,  0.09090909,  0.36363636, -0.36363636,  0.15151515,
    #          0.        ,  0.15151515]))
    #
    #  >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    #  >>> s, pvalues = asortedness__(original, original, f=wf, return_pvalues=True)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> pvalues
    #  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    #  >>> s = asortedness__(original, projected2, f=wf)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> s = asortedness__(original, projected1, f=wf)
    #  >>> min(s), max(s), s
    #  (0.878046135266, 0.99748013867, array([0.9046595 , 0.90285305, 0.93592798, 0.99748014, 0.87804614,
    #         0.98014052, 0.94867572, 0.99418203, 0.89099364, 0.92922697,
    #         0.88462681, 0.88907089]))
    # >>> s = asortedness__(original, projectedrnd, f=wf)
    #  >>> min(s), max(s), s
    #  (-0.517196452192, 0.516271063981, array([ 0.30986815, -0.15794336,  0.43126186, -0.05584362,  0.15059539,
    #          0.46072496,  0.093474  ,  0.51627106, -0.51719645, -0.12129132,
    #         -0.25956322,  0.20448257]))
    #
    #  >>> wf = partial(weightedtau, weigher=ushaped_decay_f(n=len(original)))
    #  >>> s, pvalues = asortedness__(original, original, f=wf, return_pvalues=True)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> pvalues
    #  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    #  >>> s = asortedness__(original, projected2, f=wf)
    #  >>> min(s), max(s), s
    #  (1.0, 1.0, array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    #  >>> s = asortedness__(original, projected1, f=wf)
    #  >>> min(s), max(s), s
    #  (0.795765877958, 0.983810709838, array([0.80821918, 0.79576588, 0.93524284, 0.98381071, 0.88542964,
    #         0.94271482, 0.91656289, 0.97633873, 0.89912827, 0.93150685,
    #         0.84059776, 0.89290162]))
    #  >>> s = asortedness__(original, projectedrnd, f=wf)
    #  >>> min(s), max(s), s
    #  (-0.252801992528, 0.572851805729, array([ 0.39726027, -0.03611457,  0.48069738,  0.15566625,  0.24408468,
    #          0.57285181,  0.16562889,  0.49937733, -0.25280199,  0.14445828,
    #         -0.05230386,  0.25653798]))

    X = rankdata(X, axis=0)
    X_ = rankdata(X_, axis=0)
    return sortedness(X, X_, f, return_pvalues)
