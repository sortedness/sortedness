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

from numpy import eye, argsort
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


# noinspection PyTypeChecker
def sortedness(X, X_, f=spearmanr, return_pvalues=False):  # pragma: no cover
    """
    Calculate the sortedness (a anti-stress alike correlation-based measure that ignores distance proportions) value for each point
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
        The null hypothesis is that the projection is random, i.e., sortedness = 0.5.

    Returns
    -------
        list of sortedness values (or tuples that also include pvalues)
    """

    # >>> import numpy as np
    # >>> from functools import partial
    # >>> from scipy.stats import spearmanr, weightedtau
    # >>> mean = (1, 2)
    # >>> cov = eye(2)
    # >>> rng = np.random.default_rng(seed=0)
    # >>> original = rng.multivariate_normal(mean, cov, size=12)
    # >>> s = sortedness(original, original)
    # >>> min(s), max(s), s
    # (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # >>> projected = PCA(n_components=1).fit_transform(original)
    #
    # >>> s = sortedness(original, projected)
    # >>> min(s), max(s), s
    # (0.7342657342657343, 0.9930069930069931, [0.7832167832167832, 0.7342657342657343, 0.9440559440559443, 0.9930069930069931, 0.8671328671328673, 0.9580419580419581, 0.9020979020979022, 0.9790209790209792, 0.965034965034965, 0.9790209790209792, 0.7972027972027973, 0.8881118881118882])
    # >>> from sortedness.kruskal import kruskal
    # >>> s = kruskal(original, projected, f=partial(rank_by_distances))
    # >>> 1-max(s), 1-min(s), [1 - x for x in s]
    # (0.612446612118, 0.937130538654, [0.649957653606, 0.612446612118, 0.82217831021, 0.937130538654, 0.7259583713569999, 0.846001899298, 0.764764015552, 0.8911068987040001, 0.859419610721, 0.8911068987040001, 0.661437589315, 0.748522154615])
    #
    # >>> s, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    # >>> min(s), max(s), s
    # (0.606060606060606, 0.9696969696969696, [0.6363636363636362, 0.606060606060606, 0.8484848484848484, 0.9696969696969696, 0.7575757575757575, 0.8787878787878787, 0.7878787878787877, 0.9393939393939392, 0.8787878787878787, 0.909090909090909, 0.6666666666666666, 0.7878787878787877])
    # >>> pvalues
    # [0.003181646992410881, 0.005380307706696595, 1.6342325370103148e-05, 5.010421677088344e-08, 0.00024002425044091711, 5.319397680508792e-06, 0.00010742344075677409, 3.2150205761316875e-07, 5.319397680508792e-06, 1.4655483405483405e-06, 0.0018032758136924804, 0.00010742344075677409]
    # >>> s = sortedness(original, projected)
    # >>> min(s), max(s), s
    # (0.7342657342657343, 0.9930069930069931, [0.7832167832167832, 0.7342657342657343, 0.9440559440559443, 0.9930069930069931, 0.8671328671328673, 0.9580419580419581, 0.9020979020979022, 0.9790209790209792, 0.965034965034965, 0.9790209790209792, 0.7972027972027973, 0.8881118881118882])
    # >>> s = sortedness(original, projected, f=weightedtau)
    # >>> min(s), max(s), s
    # (0.7605489568614852, 0.9876309273316981, [0.7866039053888533, 0.7605489568614852, 0.8842608200323177, 0.9876309273316981, 0.8133479034189326, 0.9364890814188078, 0.8766929005707909, 0.9737157205798583, 0.8840515688029666, 0.9231117982818149, 0.7883719725944298, 0.8196254402994617])
    # >>> wf = partial(weightedtau, weigher=lambda x: 1 / (x**2 + 1))
    # >>> s = sortedness(original, projected, f=wf)
    # >>> min(s), max(s), s
    # (0.8780461352659326, 0.9974801386703592, [0.9046594964045744, 0.9028530481822191, 0.9359279847218851, 0.9974801386703592, 0.8780461352659326, 0.9801405166364414, 0.9486757201767004, 0.994182034453877, 0.8909936435108456, 0.9292269680661559, 0.8846268149797865, 0.8890708876367106])
    # >>> projected = PCA(n_components=2).fit_transform(original)
    # >>> s = sortedness(original, projected)
    # >>> min(s), max(s), s
    # (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # >>> s = sortedness(original, projected)
    # >>> min(s), max(s), s
    # (1.0, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # >>> _, pvalues = sortedness(original, projected, return_pvalues=True)
    # >>> pvalues
    # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # >>> _, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    # >>> pvalues
    # [4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09, 4.17535139757362e-09]
    # >>> np.random.seed(0)
    # >>> projected = permutation(original)
    # >>> s = sortedness(original, projected)
    # >>> min(s), max(s), s
    # (-0.39860139860139865, 0.49650349650349657, [0.39860139860139865, -0.16783216783216784, 0.46153846153846156, 0.10489510489510491, 0.18881118881118883, 0.49650349650349657, 0.1258741258741259, 0.43356643356643365, -0.39860139860139865, 0.16783216783216784, 0.034965034965034975, 0.1258741258741259])
    # >>> # Weighted correlation-based stress can yield lower values as it focuses on higher ranks.
    # >>> s = sortedness(original, projected, f=weightedtau)
    # >>> min(s), max(s), s
    # (-0.4651020733837718, 0.45329734494008334, [0.3227605098543591, -0.1634548012060478, 0.37849108727150127, -0.005336963172840341, 0.13260504041824878, 0.40944653049836677, 0.09841254408278742, 0.45329734494008334, -0.4651020733837718, -0.014576778820393398, -0.14179941261700374, 0.1703981374526939])
    # >>> s = sortedness(original, projected, f=wf)
    # >>> min(s), max(s), s
    # (-0.51719645219205, 0.5162710639809638, [0.3098681490207809, -0.15794336281630955, 0.4312618574356535, -0.055843622226620995, 0.15059539158734384, 0.4607249573077292, 0.09347400420528493, 0.5162710639809638, -0.51719645219205, -0.12129131935470096, -0.2595632212911495, 0.20448257011752424])
    # >>> _, pvalues = sortedness(original, projected, f=wf, return_pvalues=True)
    # >>> pvalues
    # [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    # >>> _, pvalues = sortedness(original, projected, f=kendalltau, return_pvalues=True)
    # >>> pvalues
    # [0.15259045898802842, 0.5452047007776174, 0.11595087782587783, 0.7373055246579552, 0.6383612539081289, 0.11595087782587783, 0.7373055246579552, 0.11595087782587783, 0.11595087782587783, 0.5452047007776174, 1.0, 0.5452047007776174]
    # >>> projected = permutation(original)
    # >>> s = sortedness(original, projected, f=wf)
    # >>> min(s), max(s), s
    # (-0.314701609990475, 0.23829059925032403, [-0.29851544202405506, -0.26182285121352383, 0.12709476627417826, -0.1198045732618854, -0.004255962101312155, -0.09676162002121147, 0.23829059925032403, -0.1825003551612558, -0.15805497961328627, 0.204719128528062, 0.002494216275595984, -0.314701609990475])
    # >>> corrs, pvalues = sortedness(original, projected, return_pvalues=True)
    # >>> corrs
    # [0.15384615384615385, 0.027972027972027972, 0.16783216783216784, 0.02097902097902098, -0.0979020979020979, 0.1258741258741259, 0.27272727272727276, 0.09090909090909093, 0.18181818181818185, 0.2027972027972028, 0.11188811188811189, -0.013986013986013986]
    # >>> pvalues
    # [0.6330906812462618, 0.9312343512018808, 0.602099427786538, 0.9484022252365223, 0.762121655996299, 0.6966831093957659, 0.3910967709418962, 0.7787253962454419, 0.5717012385276553, 0.5273023541661082, 0.729194990751066, 0.9655902689187795]

    result, pvalues = [], []
    for a, b in zip(X, X_):
        corr, pvalue = f(euclidean__n_vs_1(X, a), euclidean__n_vs_1(X_, b))
        print("TODO: not implemented: pegar correlação(ranking_identidade, ranking_indexado_byN médio de M)")
        result.append(corr)
        pvalues.append(pvalue)
    if return_pvalues:
        return result, pvalues
        # return list(zip(result, pvalues))
    return result
