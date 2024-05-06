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


class Calmness:
    def __init__(self, X: tensor, w: tensor = None, sortbyX_=True):
        """Calmness: The opposite of stress; neighbors can be weighted by a given vector (tensor)

        >>> from torch import tensor
        >>> X = tensor([[1.,2], [3,4], [5,6]])
        >>> w = tensor([0.5, 0.3, 0.25])
        >>> Calmness(X, w)(X)
        tensor(1.)
        >>> X_ = tensor([[3.,4], [1,2], [5,6]])
        >>> Calmness(X, w)(X_)
        tensor(0.5873)

        :param X: Original data.
        :param w: Weights vector. |w| < |X|. Only the first |w| neighbors are used - for efficiency.
        :return:
        """
        if w is not None and w.shape[0] >= X.shape[0]:
            Xsize = X.shape[0]
            wsize = float(w.shape[0])
            raise Exception(f"Number of neighbors should be at most |X| - 1, i.e., {(Xsize - 1)=}; not {wsize=}")

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
        # noinspection PyUnresolvedReferences
        self.w = w #None if w is None else w.reshape(-1, w.shape[0])
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
        miniD_ = torch.cdist(miniX_, X_)[a].reshape(minin, -1)

        if self.w is None:
            # stress = torch.sum((miniD - miniD_) ** 2) / torch.sum(miniD ** 2) # normalized by all
            stress = torch.mean(torch.sum((miniD - miniD_) ** 2, dim=1) / torch.sum(miniD ** 2, dim=1))  # normalized by row
        else:
            if self.sortbyX_:
                miniDsorted_, idxs_for_miniDsorted_ = torch.topk(miniD_, self.k, largest=False, dim=1)
                miniDsorted = miniD[torch.arange(miniD.size(0)).unsqueeze(1), idxs_for_miniDsorted_]
            else:
                miniDsorted = miniD
                # miniDsorted_ =  torch.cdist(miniX_, X_[self.idxs_f,bb,b,b,mb,mor_Dsorted])[a].reshape(minin, -1)
                # index each row
                miniDsorted_ = miniD_[torch.arange(miniD_.size(0)).unsqueeze(1), self.idxs_for_Dsorted[idxs]]
            # stress = torch.sqrt(torch.sum((miniD - miniDsorted_) ** 2)) # raw
            # stress = torch.sum((miniD - miniDsorted_) ** 2 * self.w) / torch.sum(miniD ** 2 * self.w) # normalized by all
            stress = torch.mean(torch.sum((miniDsorted - miniDsorted_) ** 2 * self.w, dim=1) / torch.sum(miniDsorted ** 2 * self.w, dim=1))  # normalized by row

        return 1 - stress
