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

from sortedness.new.math_functions import softtau


class Sortedness:
    def __init__(self, X: tensor, w: tensor = None, sortbyX_=True, surrogate_correl: callable = softtau):
        """Sortedness according to transformed data when `sortbyX_=True`

        >>> from torch import tensor
        >>> X = tensor([[1.,2], [3,4], [5,6]])
        >>> w = tensor([0.5, 0.3, 0.25])
        >>> Sortedness(X, w)(X)
        tensor(1.)
        >>> X_ = tensor([[3.,4], [1,2], [5,6]])
        >>> Sortedness(X, w)(X_)
        tensor(0.5873)

        :param X: Original data.
        :param w: Weights - should have the same length as `X`.
        :return:
        """
        if w is not None and w.shape[0] >= X.shape[0]:
            Xsize = X.shape[0]
            wsize = float(w.shape[0])
            raise Exception(f"Number of neighbors should be at most |X| - 1, i.e., {(Xsize - 1)=}; not {wsize=}")
        self.surrogate_correl = surrogate_correl

        if isinstance(X, ndarray):
            X = from_numpy(X.astype(np.float32))
        self.n = X.shape[0]

        # Distance matrix without diagonal.
        # 64GiB souhld suffice for 80k points.
        # Otherwise, we can calculate miniD every step.
        self.seq = torch.arange(self.n)
        self.D = torch.cdist(X, X)[self.seq != self.seq[:, None]].reshape(self.n, -1)

        if w is not None:
            self.k = w.shape[0]
        if not sortbyX_:
            self.D, self.idxs_for_Dsorted = torch.topk(self.D, self.k, largest=False, dim=1)
        self.sortbyX_ = sortbyX_
        self.w = w  # None if w is None else w.reshape(-1, w.shape[0])
        # noinspection PyTypeChecker
        self.wtot = None if w is None else torch.sum(w)
        self.seq = torch.arange(self.n)

    def __call__(self, X_: tensor, idxs=None, **f__kwargs):
        if idxs is None:
            miniD = self.D
            miniX_ = X_
            idxs = self.seq
        else:
            miniD = self.D[idxs]
            miniX_ = X_[idxs]
        minin = len(idxs)
        a = self.seq != idxs[:, None]
        if self.sortbyX_:
            miniD_ = torch.cdist(miniX_, X_)[a].reshape(minin, -1)
            if self.w is None:
                miniDsorted = miniD
                miniDsorted_ = miniD_
            else:
                miniDsorted_, idxs_for_miniDsorted_ = torch.topk(miniD_, self.k, largest=False, dim=1)
                miniDsorted = miniD[torch.arange(miniD.size(0)).unsqueeze(1), idxs_for_miniDsorted_]
        else:
            miniDsorted = miniD
            if self.w is None:
                miniDsorted_ = torch.cdist(miniX_, X_)[a].reshape(minin, -1)
            else:
                # miniDsorted_ =  torch.cdist(miniX_, X_[self.idxs_f,bb,b,b,mb,mor_Dsorted])[a].reshape(minin, -1)
                # index each row
                miniD_ = torch.cdist(miniX_, X_)[a].reshape(minin, -1)
                b = torch.arange(miniD_.size(0)).unsqueeze(1)
                miniDsorted_ = miniD_[b, self.idxs_for_Dsorted[idxs]]
        c = s = 0
        for minidsorted, minidsorted_ in zip(miniDsorted, miniDsorted_):
            s += self.surrogate_correl(minidsorted, minidsorted_, self.w)
            c += 1
        return s / c
