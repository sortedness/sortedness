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


from math import pi

import numpy as np
from scipy.stats import weightedtau, kendalltau

from sortedness.embedding.sigmas_ import findsigma, findweight
from sortedness.local import geomean_np


def balanced_kendalltau_cauchy(unordered_values, unordered_values_, beta=0.5, gamma=4):
    """
    >>> round(balanced_kendalltau(np.array([2,1,3,4,5]), np.array([2,1,3,4,5]), beta=1), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=1), 5)
    -1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), beta=0), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=0), 5)
    -1.0
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # order of importance is defined by internally sorting the first sequence.
    >>> round(balanced_kendalltau(np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), np.array([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # weaker break of trustworthiness = an intermediate median distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of trustworthiness
    0.88332
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # strong break of continuity
    0.53172
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of continuity
    0.76555
    """
    if beta == 1:
        tau_local = 1
    else:
        idx = np.argsort(unordered_values, kind="stable")
        tau_local = weightedtau(unordered_values, unordered_values_, weigher=lambda r: 1 / pi * gamma / (gamma ** 2 + r ** 2), rank=idx)[0]
    tau_global = 1 if beta == 0 else kendalltau(unordered_values, unordered_values_)[0]
    return geomean_np(tau_local, tau_global, beta)


def balanced_kendalltau_gaussian(unordered_values, unordered_values_, alpha=0.5, beta=0.5, kappa=5, pct=90, verbose=False) -> float:
    """
    >>> round(balanced_kendalltau(np.array([2,1,3,4,5]), np.array([2,1,3,4,5]), beta=1), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=1), 5)
    -1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]), beta=0), 5)
    1.0
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5]), np.array([5,4,3,2,1]), beta=0), 5)
    -1.0
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # order of importance is defined by internally sorting the first sequence.
    >>> round(balanced_kendalltau(np.array([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), np.array([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), beta=0), 5) # strong break of trustworthiness
    0.83258
    >>> # weaker break of trustworthiness = an intermediate median distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of trustworthiness
    0.88332
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # strong break of continuity
    0.53172
    >>> round(balanced_kendalltau(np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), np.array([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), beta=0), 5) # weaker break of continuity
    0.76555
    """
    if beta == 1:
        lo = lo_ = 1
    else:
        sigma = findsigma(pct, kappa)
        w = [findweight(x, sigma) for x in range(len(unordered_values))]

        if alpha < 1:
            if verbose:
                print("Indexing based on X...", flush=True, end="\t")
            idx = np.argsort(unordered_values, kind="stable")
            if verbose:
                print("Calculating local measure weighted by X...", flush=True, end="\t")
            lo = (weightedtau(unordered_values, unordered_values_, weigher=lambda x: w[x], rank=idx)[0] + 1) / 2
            if verbose:
                print(f"OK: {lo}", flush=True)
        else:
            lo = 1

        if alpha > 0:
            if verbose:
                print("Indexing based on X_...", flush=True, end="\t")
            idx_ = np.argsort(unordered_values_, kind="stable")
            if verbose:
                print("Calculating local measure weighted by X_...", flush=True, end="\t")
            lo_ = (weightedtau(unordered_values, unordered_values_, weigher=lambda x: w[x], rank=idx_)[0] + 1) / 2
            if verbose:
                print(f"OK: {lo_}", flush=True)
        else:
            lo_ = 1

    g = (kendalltau(unordered_values, unordered_values_)[0] + 1) / 2 if beta > 0 else 1
    z = alpha * beta - alpha
    s = lo ** (z - beta + 1) * lo_ ** (-z) * g ** beta
    return 2 * s - 1
