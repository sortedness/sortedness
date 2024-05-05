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


import torch
from torch import tanh, sum, sqrt, abs, tensor


def dist2prob(D):
    """Transform distances to probabilities of proximity

    Let D, P be the input, output vectors for `dist2prob`:
        max(D)  →   p=0
        d=0     →   max(P)
    """
    Drev = torch.max(D, dim=1, keepdim=True)[0] - D
    s = torch.sum(Drev, dim=1, keepdim=True)
    return Drev / (s + 0.00000001)


def pdiffs(x):
    dis = x.unsqueeze(1) - x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def psums(x):
    dis = x.unsqueeze(1) + x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def pmuls(x):
    dis = x.unsqueeze(1) * x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def softtau(a, b, w=None, smoothness=1):
    """
    >>> from torch import tensor
    >>> from scipy.stats import kendalltau
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), .01)
    tensor(1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), .01)
    tensor(-1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), 2)
    tensor(0.7473)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), 2)
    tensor(-0.7473)
    >>> round(float(softtau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .000001)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
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
    """
    da, db = pdiffs(a), pdiffs(b)
    ta, tb = tanh(da / smoothness), tanh(db / smoothness)
    if w is None:
        num = sum(ta * tb)
        v = sum(abs(ta)) * sum(abs(tb))
    else:
        sw = psums(w)
        num = sum(ta * tb * sw)
        v = sum(abs(ta * sw)) * sum(abs(tb * sw))
    den = sqrt(v + 0.00000000001)
    return num / den


def relative_stress(a, b, w=None):
    """"""
    da, db = pdiffs(a), pdiffs(b)
    if w is None:
        return torch.sum((da - db) ** 2) / torch.sum(da ** 2)
    else:
        sw = psums(w)
        return torch.sum((da - db) ** 2 * sw) / torch.sum(da ** 2 * sw)
