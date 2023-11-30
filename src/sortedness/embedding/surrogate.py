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
import math
from itertools import repeat

import torch
from scipy.stats import weightedtau, kendalltau
from torch import tanh, sum, topk, sqrt, abs, tensor

from sortedness.local import geomean_np


# lin = lambda r,k: 2*(k+1-i)/k*(k+1)
def cau(r, gamma=4):
    """
    >>> import torch
    >>> from lange import ap
    >>> cau(torch.tensor(ap[0,1,...,2]))
    [1.0, 0.606530659713, 0.135335283237, 0.011108996538, 0.000335462628, 3.726653e-06, 1.523e-08, 2.3e-11, 0.0, 0.0]
    """

    return gamma ** 2 / (gamma ** 2 + r ** 2)


har = lambda r: 1 / (r + 1)


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


# tanh = lambda x: torch.max(tensor([-1]), torch.min(tensor([1]), x))


def surrogate_tau(a, b, smoothness):
    """
    >>> from torch import tensor
    >>> from scipy.stats import kendalltau
    >>> surrogate_tau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]), .01)
    tensor(1.)
    >>> surrogate_tau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]), .01)
    tensor(-1.)
    >>> round(float(surrogate_tau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]), .01)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
    0.948683
    """
    da, db = pdiffs(a), pdiffs(b)
    ta, tb = tanh(da / smoothness), tanh(db / smoothness)
    num = sum(ta * tb)
    v = sum(abs(ta)) * sum(abs(tb))
    den = sqrt(v + 0.00000000001)
    return num / den


def surrogate_wtau(a, b, w, smoothness):
    """
    >>> from torch import tensor
    >>> from scipy.stats import kendalltau
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), .01)
    tensor(1.)
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), .01)
    tensor(-1.)
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), 2)
    tensor(0.7473)
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), 2)
    tensor(-0.7473)
    >>> round(float(surrogate_wtau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .000001)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
    0.948683

    >>> cau = torch.tensor([0.07961783439490445, 0.07493443237167478, 0.06369426751592357, 0.05095541401273885, 0.03980891719745223, 0.031070374398011493, 0.02449779519843214, 0.019598236158745713, 0.01592356687898089, 0.013132838663077023, 0.010981770261366132, 0.00929843321400344, 0.007961783439490446, 0.006885866758478223, 0.006008893161879581])
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(float(surrogate_wtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), cau, 0.00001)), 5)
    0.83258
    >>> # order of importance is defined by weights, not by values of argument `a`.
    >>> round(float(surrogate_wtau(torch.tensor([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), torch.tensor([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), cau, 0.00001)), 5) # strong break of trustworthiness
    0.53172
    >>> # weak break of trustworthiness = an intermediate distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(float(surrogate_wtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # a weaker break of trustworthiness
    0.88332
    >>> round(float(surrogate_wtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # strong break of continuity
    0.53172
    >>> round(float(surrogate_wtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # weaker break of continuity
    0.76555
    """
    da, db, sw = pdiffs(a), pdiffs(b), psums(w)
    ta, tb = tanh(da / smoothness), tanh(db / smoothness)
    num = sum(ta * tb * sw)
    v = sum(abs(ta * sw)) * sum(abs(tb * sw))
    den = sqrt(v + 0.00000000001)
    return num / den


"""
000000001:	optimized sur: 0.2526  local/globa: 0.2526 0.0000  REF: 0.3930 0.2909		0.300000
000000002:	optimized sur: 0.3596  local/globa: 0.3596 0.0000  REF: 0.5666 0.2984		0.300000
000000003:	optimized sur: 0.3662  local/globa: 0.3662 0.0000  REF: 0.5824 0.2662		0.300000
000000004:	optimized sur: 0.3639  local/globa: 0.3639 0.0000  REF: 0.5574 0.2855		0.300000
000000005:	optimized sur: 0.3627  local/globa: 0.3627 0.0000  REF: 0.5660 0.2971		0.300000

000000001:	optimized sur: 0.2526  local/globa: 0.2526 0.0000  REF: 0.3930 0.2909		0.300000
000000002:	optimized sur: 0.3596  local/globa: 0.3596 0.0000  REF: 0.5666 0.2984		0.300000
000000003:	optimized sur: 0.3662  local/globa: 0.3662 0.0000  REF: 0.5824 0.2662		0.300000
000000004:	optimized sur: 0.3639  local/globa: 0.3639 0.0000  REF: 0.5574 0.2855		0.300000
000000005:	optimized sur: 0.3627  local/globa: 0.3627 0.0000  REF: 0.5660 0.2971		0.300000

"""


# def surrogate_tau_rel(a, b, smoothness):
#     """
#     >>> from torch import tensor
#     >>> from scipy.stats import kendalltau
#     >>> surrogate_tau_rel(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]), .00001)
#     tensor(1.0000)
#     >>> surrogate_tau_rel(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]), .00001)
#     tensor(-1.0000)
#     >>> round(float(surrogate_tau_rel(tensor([1,2,2,4,5]), tensor([5,4,3,2,1]), .00001)), 4)
#     -0.9487
#     >>> round(float(surrogate_tau_rel(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]), .00001)), 4)
#     0.9487
#     >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 4)
#     0.9487
#     >>> surrogate_tau_rel(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]), 0.9)
#     tensor(0.4734)
#     >>> surrogate_tau_rel(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]), 0.9)
#     tensor(-0.4357)
#     """
#     # reminder: relative is more optimistic here due to integer ties
#     da, db, sa, sb = pdiffs(a), pdiffs(b), psums(a), psums(b)
#     ta = torch.sign(da) * (abs(da + 0.0000000001) / (sa + 0.0000000001)) ** smoothness
#     tb = torch.sign(db) * (abs(db + 0.0000000001) / (sb + 0.0000000001)) ** smoothness
#     num = sum(ta * tb)
#     den = sqrt(sum(abs(ta)) * sum(abs(tb)) + 0.000000000001)
#     return num / den
#
#
# def surrogate_wtau_rel(a, b, w, smoothness):
#     """
#     >>> from torch import tensor
#     >>> from scipy.stats import kendalltau
#     >>> surrogate_wtau_rel(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), .00001)
#     tensor(1.0000)
#     >>> surrogate_wtau_rel(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), .00001)
#     tensor(-1.0000)
#     >>> surrogate_wtau_rel(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), 4)
#     tensor(0.1387)
#     >>> surrogate_wtau_rel(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), 4)
#     tensor(-0.1071)
#     >>> round(float(surrogate_wtau_rel(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .000001)), 6)
#     0.948682
#     >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
#     0.948683
#     """
#     da, db, sa, sb, sw = pdiffs(a), pdiffs(b), psums(a), psums(b), psums(w)
#     ta = torch.sign(da) * (abs(da + 0.0000000001) / (sa + 0.0000000001)) ** smoothness
#     tb = torch.sign(db) * (abs(db + 0.0000000001) / (sb + 0.0000000001)) ** smoothness
#     num = sum(ta * tb * sw)
#     v = sum(abs(ta * sw)) * sum(abs(tb * sw))
#     den = sqrt(v + 0.000000000001)
#     return num / den


def geomean(lo, gl, beta=0.5):
    """
    >>> from torch import tensor
    >>> geomean(tensor([0.6]), tensor([0.64]))
    tensor([0.6199])
    """
    l = (lo + 1) / 2
    g = (gl + 1) / 2
    return torch.exp((1 - beta) * torch.log(l + 0.00000000001) + beta * torch.log(g + 0.00000000001)) * 2 - 1


def loss_function(miniD, miniD_, miniDsorted, miniidxs_by_D, k, global_k, w, alpha=0.5, beta=0.5, smoothness_tau=1, min_global_k=100, max_global_k=1000, ref=False):
    n, v = miniD.shape  # REMINDER: n is the size of the minibatch
    if global_k == "sqrt":
        global_k = max(min_global_k, min(max_global_k, int(math.sqrt(v))))
    if global_k > v:
        global_k = v
    if k + 1 > v:
        k = v - 1
    if k < 1:
        raise Exception(f"`k` must be greater than 1: {k} > 1")
    if global_k < 1:
        raise Exception(f"`global_k` must be greater than 1: {global_k} > 1")
    if not (0 <= alpha <= 1):
        raise Exception(f"`alpha` outside valid range: 0 <= {alpha} <= 1")
    if not (0 <= beta <= 1):
        raise Exception(f"`beta` outside valid range: 0 <= {beta} <= 1")
    if not (0.00001 <= smoothness_tau <= 100):
        raise Exception(f"`smoothness_tau` outside valid range: 0.0001 <= {smoothness_tau} <= 100")

    mu = mu_local_acc = mu_global_acc = tau_local_acc = tau_global_acc = 0
    rnd_idxs = torch.randperm(v)
    start = 0
    if alpha == 1:
        miniDsorted, miniidxs_by_D = repeat(None), repeat(None)
    for d, d_, dsorted, idxs_by_D in zip(miniD, miniD_, miniDsorted, miniidxs_by_D):
        # local
        if beta < 1:
            if 0 < alpha < 1:
                a0, b0 = dsorted, d_[idxs_by_D]
                mu_local_d = surrogate_wtau(a0, b0, w[:k], smoothness_tau)

                a1, idxs_by_D_ = topk(d_, k, largest=False)
                b1 = d[idxs_by_D_]
                mu_local_d_ = surrogate_wtau(a1, b1, w[:k], smoothness_tau)

                mu_local = geomean(mu_local_d, mu_local_d_, alpha)
            else:
                if alpha == 0:
                    a0, b0 = dsorted, d_[idxs_by_D]  # todo: não precisa reindexar toda hora, só reindexar X do lado de fora conforme D
                    mu_local = surrogate_wtau(a0, b0, w[:k], smoothness_tau)  # todo: não precisa recalcular step para a0 toda hora
                else:
                    a1, idxs_by_D_ = topk(d_, k, largest=False)
                    b1 = d[idxs_by_D_]
                    mu_local = surrogate_wtau(a1, b1, w[:k], smoothness_tau)
            mu_local_acc += mu_local

        # global
        if beta > 0:
            end = start + global_k
            if end > v:
                start = 0
                end = global_k
                rnd_idxs = torch.randperm(v)
            gidxs = rnd_idxs[start:end]
            start += global_k
            ga = d[gidxs]
            gb = d_[gidxs]
            # mu_global = tensor(kendalltau(ga, gb)[0])
            mu_global = surrogate_tau(ga, gb, smoothness_tau)
            mu_global_acc += mu_global

        if 0 < beta < 1:
            mu += geomean(mu_local, mu_global, beta)
        elif beta == 0:
            mu += mu_local
        else:
            mu += mu_global

        if ref:
            # todo: ref is not perfect as it is sampled/shortened
            if alpha == 0 or beta == 1:
                a1, idxs_by_D = topk(d, 2 * k, largest=False)
                b1 = d_[idxs_by_D]
            if alpha == 1 or beta == 1:
                b0, idxs_by_D_ = topk(d_, 2 * k, largest=False)
                a0 = d[idxs_by_D_]
            lo0 = weightedtau(a0.cpu().detach().numpy(), b0.cpu().detach().numpy(), weigher=lambda r: w[r], rank=False, additive=True)[0]
            lo1 = weightedtau(a1.cpu().detach().numpy(), b1.cpu().detach().numpy(), weigher=lambda r: w[r], rank=False, additive=True)[0]
            tau_local_acc += geomean_np(lo0, lo1, alpha)

            p, t = d.cpu().detach().numpy(), d_.cpu().detach().numpy()
            tau_global_acc += kendalltau(p, t)[0]

    return mu / n, mu_local_acc / n, mu_global_acc / n, tau_local_acc / n, tau_global_acc / n
