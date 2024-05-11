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


def calmness(D, D_, w):
    # stress = torch.sqrt(torch.sum((miniD - miniDsorted_) ** 2)) # raw
    # stress = torch.sum((miniD - miniDsorted_) ** 2 * self.w) / torch.sum(miniD ** 2 * self.w) # normalized by all
    if w is None:
        return 1 - torch.mean(torch.sum((D - D_) ** 2, dim=1) / torch.sum(D ** 2, dim=1))  # normalized by row
    return 1 - torch.mean(torch.sum((D - D_) ** 2 * w, dim=1) / torch.sum(D ** 2 * w, dim=1))  # normalized by row


def transitiveness(a, b=None, w=None, lambd=1.0):
    """
    >>> import torch
    >>> transitiveness(torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), lambd=1)
    tensor(1.)
    >>> transitiveness(torch.tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]]), torch.tensor([[1,2,3],[3,4,5],[5,6,7],[8,7,9]]), lambd=0.01)
    tensor(0.7500)

    :param a:
    :param b:
    :param w:
    :param lambd:
    :return:
    """
    ta = torch.tanh((a[:, 1:] - a[:, :-1]) / lambd)
    tb = torch.tanh((b[:, 1:] - b[:, :-1]) / lambd)
    agreem = ta if b is None else ta * tb
    if w is None:
        w = 1
    else:
        w = w[:-1]
        agreem *= w
    res = agreem.sum(dim=1) / (ta ** 2 * w).sum(dim=1)
    return torch.mean(res)
