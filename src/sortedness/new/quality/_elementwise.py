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
from torch import tensor

from sortedness.new.quality.abs import Quality


class Elementwise(Quality):
    def __init__(self, measure: callable, X: tensor, w: tensor = None, sortbyX_=True):
        """Quality surrogate function for elementwise measures

        :param measure:
        :param X: Original data.
        :param w: Weights vector. |w| < |X|. Only the first |w| neighbors are used - for efficiency.
        :param sortbyX_: If `True`, sort according to transformed data (X_, instead of X).
        :return:
        """
        super().__init__(X, w, sortbyX_, measure)

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
            miniDsorted, miniDsorted_ = miniD, miniD_
        else:
            if self.sortbyX_:
                miniDsorted_, idxs_for_miniDsorted_ = torch.topk(miniD_, self.k, largest=False, dim=1)
                miniDsorted = miniD[torch.arange(miniD.size(0)).unsqueeze(1), idxs_for_miniDsorted_]
            else:
                miniDsorted = miniD
                # index each row
                miniDsorted_ = miniD_[torch.arange(miniD_.size(0)).unsqueeze(1), self.idxs_for_Dsorted[idxs]]
        return self.measure(miniDsorted, miniDsorted_, self.w)
