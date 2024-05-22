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


def calmness(D, D_, w=None):
    # stress = torch.sqrt(torch.sum((miniD - miniDsorted_) ** 2)) # raw
    # stress = torch.sum((miniD - miniDsorted_) ** 2 * self.w) / torch.sum(miniD ** 2 * self.w) # normalized by all
    if w is None:
        return 1 - 2 * torch.mean(torch.sum((D - D_) ** 2, dim=1) / torch.sum(D ** 2, dim=1))  # normalized by row
    return 1 - 2 * torch.mean(torch.sum((D - D_) ** 2 * w, dim=1) / torch.sum(D ** 2 * w, dim=1))


def transitiveness(a, b=None, w=None, lambd=1.0):
    """
    Measure order assuming rows are almost ordered

    A major limitation, and distinction from a rank correlation index, can be seen in the first two examples below.
    Only adjacent items are compared.

    todo: can be used at later stages of DR

    >>> import torch
    >>> transitiveness(torch.tensor([[1,2,3,4,5,6]]), torch.tensor([[1,2,4,3,5,6]]), lambd=0.001)
    tensor(0.6000)
    >>> transitiveness(torch.tensor([[1,2,3,4,5,6]]), torch.tensor([[4,5,6,1,2,3]]), lambd=0.001)
    tensor(0.6000)
    >>> transitiveness(torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), lambd=1)
    tensor(1.)
    >>> transitiveness(torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), torch.tensor([[1,2,3],[3,4,5],[5,6,7],[8,7,9]]), lambd=0.01)
    tensor(0.7500)
    >>> transitiveness(torch.tensor([[1,2,3,4,5,6,7]]), torch.tensor([[1,2,3,4,5,7,6]]), lambd=0.01)
    tensor(0.6667)
    >>> from sortedness.new.weighting import gaussian, cauchy
    >>> cauw = cauchy(7, kappa=5, pct=90)
    >>> round(float(transitiveness(torch.tensor([[1,2,3,4,5,6,7]]), torch.tensor([[7,1,2,3,4,5,6]]), w=cauw)), 8)
    -0.3713732
    >>> round(float(transitiveness(torch.tensor([[1,2,3,4,5,6,7]]), torch.tensor([[2,3,4,5,6,7,1]]), w=cauw)), 8)
    0.95229307
    >>> round(float(transitiveness(torch.tensor([[1,2,3,4,5,6,7]]), torch.tensor([[7,6,5,4,3,2,1]]), w=cauw)), 6)
    -1.0

    :param a:
    :param b:
    :param w:
    :param lambd:
    :return:
    """
    ta = torch.tanh((a[:, 1:] - a[:, :-1]) / lambd)
    tb = torch.tanh((b[:, 1:] - b[:, :-1]) / lambd)
    num = ta if b is None else ta * tb
    if w is None:
        w = 1
    else:
        w = w[:-1]
        num *= w
    den = (ta ** 2 * w).sum(dim=1) if b is None else (ta ** 2 * w).sum(dim=1) * (tb ** 2 * w).sum(dim=1)
    res = num.sum(dim=1) / torch.sqrt(den + .00000001)
    return torch.mean(res)
