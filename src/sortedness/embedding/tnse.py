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


def pd(x):
    dis = x.unsqueeze(1) - x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def surrogate_tau(a, b):
    da, db = pd(a), pd(b)
    return sum(sigmoid(da * db))


def lossf(predicted_D, expected_D, i=None, running=None):
    n = predicted_D.shape[0] - 1
    r = []
    mu = wtau = 0
    for pred, target in zip(predicted_D, expected_D):  #tODO: same size? ver diagonal
        surr = surrogate_tau(pred.view(n), target.view(n))
        wtau += weightedtau(pred.detach().cpu().numpy(), target.detach().cpu().numpy())[0]
        mu += surr
    plt.title(f"{i}:    {wtau / predicted_D.shape[0]:.8f}    {float(mu):.8f}   {running=}", fontsize=20)
    return -mu / (n + 1)
