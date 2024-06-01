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
import numpy as np
import torch
from numpy import ndarray
from torch import tensor, from_numpy


class Quality:
    def __init__(self, X: tensor, w: tensor, sortbyX_, measure: callable, skip_dist_calculation=False):
        from sortedness.new.quality._nodist import NoDist
        if w is not None and w.shape[0] >= X.shape[0] and not isinstance(self, NoDist):
            Xsize = X.shape[0]
            wsize = float(w.shape[0])
            raise Exception(f"Number of neighbors should be at most |X| - 1, i.e., {(Xsize - 1)=}; not {wsize=}")
        self.measure = measure

        if isinstance(X, ndarray):
            X = from_numpy(X.astype(np.float32))
        self.n, self.m = X.shape

        self.w = w  # None if w is None else w.reshape(-1, w.shape[0])
        # noinspection PyTypeChecker
        self.wtot = None if w is None else torch.sum(w)
        # noinspection PyUnresolvedReferences
        self.k = self.n if w is None else w.shape[0]
        self.seq = torch.arange(self.n)

        if skip_dist_calculation:
            return X

        # Distance matrix without diagonal.
        # 64GiB souhld suffice for 80k points.
        # Otherwise, we can calculate miniD every step.
        self.seq = torch.arange(self.n)
        self.D = torch.cdist(X, X)[self.seq != self.seq[:, None]].reshape(self.n, -1)

        if not sortbyX_:
            self.D, self.idxs_for_Dsorted = torch.topk(self.D, self.k, largest=False, dim=1)
        self.sortbyX_ = sortbyX_

    def __call__(self, X_: tensor, idxs=None, **f__kwargs):
        raise NotImplementedError

    def check(self, other):
        from sortedness.new.quality._pairwise import Pairwise
        from sortedness.new.quality._elementwise import Elementwise
        from sortedness.new.quality._nodist import NoDist
        if isinstance(self, (Pairwise, NoDist)) and isinstance(other, Elementwise) or isinstance(self, Elementwise) and isinstance(other, (Pairwise, NoDist)):
            raise Exception(f"Cannot mix Pairwise/NoDist {type(self)} and Elementwise {type(other)} quality surrogate functions.")

    def __add__(self, other: 'Quality'):
        if isinstance(other, Quality):
            f = lambda *args, **kwargs: self(*args, **kwargs) + other(*args, **kwargs)
        else:
            f = lambda *args, **kwargs: self(*args, **kwargs) + other
        return Expr(f)

    def __sub__(self, other: 'Quality'):
        if isinstance(other, Quality):
            f = lambda *args, **kwargs: self(*args, **kwargs) - other(*args, **kwargs)
        else:
            f = lambda *args, **kwargs: self(*args, **kwargs) - other
        return Expr(f)

    def __mul__(self, other: 'Quality'):
        if isinstance(other, Quality):
            f = lambda *args, **kwargs: self(*args, **kwargs) * other(*args, **kwargs)
        else:
            f = lambda *args, **kwargs: self(*args, **kwargs) * other
        return Expr(f)

    def __truediv__(self, other: 'Quality'):
        if isinstance(other, Quality):
            f = lambda *args, **kwargs: self(*args, **kwargs) / other(*args, **kwargs)
        else:
            f = lambda *args, **kwargs: self(*args, **kwargs) / other
        return Expr(f)

    def __pow__(self, other: 'Quality'):
        if isinstance(other, Quality):
            f = lambda *args, **kwargs: self(*args, **kwargs) ** other(*args, **kwargs)
        else:
            f = lambda *args, **kwargs: self(*args, **kwargs) ** other
        return Expr(f)

    def __rmul__(self, other: 'Quality'):
        f = lambda *args, **kwargs: other * self(*args, **kwargs)
        return Expr(f)

    def __rtruediv__(self, other: 'Quality'):
        f = lambda *args, **kwargs: other / self(*args, **kwargs)
        return Expr(f)

    def __radd__(self, other: 'Quality'):
        f = lambda *args, **kwargs: other + self(*args, **kwargs)
        return Expr(f)

    def __rsub__(self, other: 'Quality'):
        f = lambda *args, **kwargs: other - self(*args, **kwargs)
        return Expr(f)

    def __rpow__(self, other: 'Quality'):
        f = lambda *args, **kwargs: other ** self(*args, **kwargs)
        return Expr(f)


class Expr(Quality):
    # noinspection PyMissingConstructor
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)
