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


def balanced_ab(l0, l1, g, beta, z):
    """geomean
    z = alpha * beta - alpha"""
    l0 = (l0 + 1) / 2
    l1 = (l1 + 1) / 2
    g = (g + 1) / 2
    return 2 * l0 ** (z - beta + 1) * l1 ** (-z) * g ** beta - 1


def balanced_abt(l0, l1, g, beta, theta, z):
    """generalized geomean

    :param theta:  balances between permissiveness or strictness
                    intended to attenuate the effect of a very good component compensating a bad one
    :param z: `alpha * beta - alpha`
    :return:
    """
    l0 = (l0 + 1) / 2
    l1 = (l1 + 1) / 2
    g = (g + 1) / 2
    r = (z - beta + 1) * l0 ** (theta + 1) - z * l1 ** (theta + 1) + beta * g(theta + 1)
    return 2 * r ** (1 / theta + 1) - 1
