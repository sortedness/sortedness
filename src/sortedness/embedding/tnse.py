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

import matplotlib.pyplot as plt
import torch
from scipy.stats import weightedtau
from torch import sigmoid, sum


def pdiffs(x):
    dis = x.unsqueeze(1) - x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]

def psums(x):
    dis = x.unsqueeze(1) + x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def surrogate_tau(a, b):
    da, db = pdiffs(a), pdiffs(b)
    return sum(sigmoid(da * db))


def surrogate_wtau(a, b, w):
    da, db, sw = pdiffs(a), pdiffs(b), psums(w)
    return sum(sigmoid(da * db) * sw)


def lossf(predicted_D, expected_D, i=None, running=None):
    n = predicted_D.shape[0]
    m = predicted_D.shape[1]
    r = []
    mu = wtau = 0
    for pred, target in zip(predicted_D, expected_D):  # tODO: same size? ver diagonal
        surr = surrogate_tau(pred.view(m), target.view(m))
        wtau += weightedtau(pred.detach().cpu().numpy(), target.detach().cpu().numpy())[0]
        mu += surr
    plt.title(f"{i}:    {wtau / predicted_D.shape[0]:.8f}    {float(mu):.8f}   {running=}", fontsize=20)
    return -mu / n


def wlossf(predicted_D, expected_D, i=None, running=None):
    raise Exception(f"")


def lossf2(predicted_ranks, expected_ranks, i=None, running=None):
    n, o = predicted_ranks.shape
    mu = 0
    for pred, target in zip(predicted_ranks, expected_ranks):
        mu += surrogate_tau(pred, target)
    return -mu / n


def wlossf2(predicted_ranks, expected_ranks, w):
    n, o = predicted_ranks.shape
    mu = 0
    for pred, target in zip(predicted_ranks, expected_ranks):
        mu += surrogate_wtau(pred, target, w)
    return -mu / n
