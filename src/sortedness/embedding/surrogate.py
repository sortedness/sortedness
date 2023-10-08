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

import torch
from scipy.stats import weightedtau, kendalltau
from torch import tanh, sum, topk

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
    da, db = pdiffs(a), pdiffs(b)
    s = tanh(da / smooothness) * tanh(db / smooothness)
    # return sum(s) / den  # den = (o ** 2 - o) * 2
    return sum(s) / sum(torch.abs(s))  # trimm ties; for some reason the value is closer to the target function result


def surrogate_wtau(a, b, w, smooothness):
    da, db, sw = pdiffs(a), pdiffs(b), psums(w)
    s = tanh(da / smooothness) * tanh(db / smooothness)
    r = s * sw
    return sum(r) / sum(sw)  # do not trimm ties; for some reason the value is closer to the target function result
    # return sum(r) / sum(torch.abs(r))


def geomean(lo, gl, alpha=0.5):
    """
    >>> from torch import tensor
    >>> geomean(tensor([0.6]), tensor([0.64]))
    tensor([0.6199])
    """
    l = (lo + 1) / 2
    g = (gl + 1) / 2
    return torch.exp((1 - alpha) * torch.log(l) + alpha * torch.log(g)) * 2 - 1


def loss_function(predicted_D, expected_R, k, w, alpha=0.5, smooothness_tau=1, ref=False):
    n, o = predicted_D.shape
    mu = mu_local = mu_global = tau_local = tau_global = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        a, idxs = topk(pred_d, k, largest=False)
        b = target_r[idxs]
        mu_local += (mu_local0 := surrogate_wtau(a, b, w[:k], smooothness_tau))
        mu_global += (mu_global0 := surrogate_tau(pred_d, target_r, smooothness_tau))
        mu += geomean(mu_local0, mu_global0, alpha)

        if ref:
            p, t = pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy()
            tau_local += weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=lambda r: w[r], rank=False)[0]
            tau_global += kendalltau(p, t)[0]

    return mu / n, mu_local / n, mu_global / n, tau_local / n, tau_global / n
