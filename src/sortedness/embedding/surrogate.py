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
from math import pi

import torch
from scipy.stats import weightedtau, kendalltau
from torch import tanh, sum, topk, sqrt, abs

cau = lambda r, gamma=1: 1 / pi * gamma / (gamma ** 2 + r ** 2)
har = lambda r: 1 / (r + 1)


def pdiffs(x):
    dis = x.unsqueeze(1) - x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def psums(x):
    dis = x.unsqueeze(1) + x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def surrogate_tau(a, b, smooothness):
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
    ta, tb = tanh(da / smooothness), tanh(db / smooothness)
    r = ta * tb
    return sum(r) / sqrt(sum(abs(ta)) * sum(abs(tb)))


def surrogate_wtau(a, b, w, smooothness):
    """
    >>> from torch import tensor
    >>> from scipy.stats import kendalltau
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]), .01)
    tensor(1.)
    >>> surrogate_wtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]), .01)
    tensor(-1.)
    >>> round(float(surrogate_wtau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .000001)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
    0.948683
    """
    da, db, sw = pdiffs(a), pdiffs(b), psums(w)
    ta, tb = tanh(da / smooothness), tanh(db / smooothness)
    r = ta * tb * sw
    return sum(r) / sqrt(sum(abs(ta * sw)) * sum(abs(tb * sw)))


def geomean(lo, gl, beta=0.5):
    """
    >>> from torch import tensor
    >>> geomean(tensor([0.6]), tensor([0.64]))
    tensor([0.6199])
    """
    l = (lo + 1) / 2
    g = (gl + 1) / 2
    return torch.exp((1 - beta) * torch.log(l) + beta * torch.log(g)) * 2 - 1


def loss_function(predicted_D, expected_D, k, global_k, w, beta=0.5, smooothness_tau=1, min_global_k=100, max_global_k=1000, ref=False):
    n, v = predicted_D.shape
    if global_k == "sqrt":
        global_k = max(min_global_k, min(max_global_k, int(math.sqrt(v))))
    if global_k > v:
        global_k = v

    mu = mu_local = mu_global = tau_local = tau_global = 0
    rnd_idxs = torch.randperm(v)
    start = 0
    for pred_d, target_d in zip(predicted_D, expected_D):
        # REMINDER: o próprio ponto vai ser usado/amostrado 1 vez como sendo vizinho de si mesmo, não vale o custo de remover:
        # Impacta significado da medida em datasets muito pequenos, deixando ela marginalmente mais otimista
        # Pra global não impacta otimização, pra local fiz k+1 e slicing abaixo pra não perder o primeiro peso de w.

        # local
        a, idxs = topk(pred_d, k + 1, largest=False)
        a, idxs = a[1:], idxs[1:]
        b = target_d[idxs]
        mu_local += (mu_local0 := surrogate_wtau(a, b, w[:k], smooothness_tau))

        # global
        end = start + global_k
        if end > v:
            start = 0
            end = global_k
            rnd_idxs = torch.randperm(v)
        gidxs = rnd_idxs[start:end]
        ga = pred_d[gidxs]
        gb = target_d[gidxs]
        # ga = pred_d
        # gb = target_d
        mu_global += (mu_global0 := surrogate_tau(ga, gb, smooothness_tau))
        start += global_k

        mu += geomean(mu_local0, mu_global0, beta)
        if ref:
            tau_local += weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=lambda r: w[r], rank=False)[0]
            p, t = ga.cpu().detach().numpy(), gb.cpu().detach().numpy()
            tau_global += kendalltau(p, t)[0]

    return mu / n, mu_local / n, mu_global / n, tau_local / n, tau_global / n
