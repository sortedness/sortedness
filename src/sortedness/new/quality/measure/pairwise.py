#  Copyright (c) 2024. Davi Pereira dos Santos
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
from torch import tanh, sqrt, sum

from sortedness.new.math_functions import pdiffs, psums


def softtau(a, b, w=None, lambd=1.0):
    """
    >>> from torch import tensor
    >>> import torch
    >>> from scipy.stats import kendalltau
    >>> from scipy.stats import weightedtau
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]))
    tensor(1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]))
    tensor(-1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]))
    tensor(1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]))
    tensor(-1.)
    >>> round(float(softtau(tensor([1,2,3,4,5]), tensor([1,3,2,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.8
    >>> round(float(softtau(tensor([1,2,3,4,5]), tensor([1,2,2,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.948683
    >>> round(float(softtau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
    0.948683
    >>> round(kendalltau([1,2,3,4,5], [1,2,2,4,5])[0], 6)
    0.948683


    >>> cau = torch.tensor([0.07961783439490445, 0.07493443237167478, 0.06369426751592357, 0.05095541401273885, 0.03980891719745223, 0.031070374398011493, 0.02449779519843214, 0.019598236158745713, 0.01592356687898089, 0.013132838663077023, 0.010981770261366132, 0.00929843321400344, 0.007961783439490446, 0.006885866758478223, 0.006008893161879581])
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), cau, 0.00001)), 5)
    0.83258
    >>> # order of importance is defined by weights, not by values of argument `a`.
    >>> round(float(softtau(torch.tensor([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), torch.tensor([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), cau, 0.00001)), 5) # strong break of trustworthiness
    0.53172
    >>> # weak break of trustworthiness = an intermediate distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # a weaker break of trustworthiness
    0.88332
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # strong break of continuity
    0.53172
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # weaker break of continuity
    0.76555
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,2,4,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # weaker break of continuity
    0.98904

    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.83258
    >>> round(weightedtau([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], [0,14,13,12,11,10,9,8,7,6,5,4,3,2,1], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.53172
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.88332
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.53172
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,17,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.76555
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,2,4,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.98904
    """
    da, db = pdiffs(a), pdiffs(b)
    ta, tb = tanh(da / lambd), tanh(db / lambd)
    if w is None:
        num = sum(ta * tb)
        den = sum(ta ** 2) * sum(tb ** 2)
    else:
        sw = psums(w)
        num = sum(ta * tb * sw)
        den = sum(ta ** 2 * sw) * sum(tb ** 2 * sw)
    return num / sqrt(den + .00000001)


# TODO: kendall tau seems wrong, at least in scipy implementation or even the original formulation itself.
#   A tie on x without a tie on y should add zero do agreement keeping a weight in the denominator, i.e., subtract half a combination from 1.
#   For instance, for 10 combinations (n=5), it would cost 0.1, instead of 0.051317.


def relative_calmness(a, b, w=None):
    """"""
    da, db = pdiffs(a), pdiffs(b)
    if w is None:
        return 1 - sum((da - db) ** 2) / sum(da ** 2)
    else:
        sw = psums(w)
        return 1 - sum((da - db) ** 2 * sw) / sum(da ** 2 * sw)
