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
from torch import tensor, randperm, manual_seed, cat

from sortedness.new.quality.abs import Quality


class NoDist(Quality):
    def __init__(self, measure: callable, X: tensor, w: tensor = None, k=100, seed=0):
        X = super().__init__(X, w, None, measure, skip_dist_calculation=True)
        if self.w.shape[0] != X.shape[0]:
            raise Exception(f"|X| != |w|: {len(X)} != {len(w)}")
        self.X = X
        self.idxs_neighbors = randperm(self.n)
        self.idxs_start = 0
        manual_seed(seed)
        self.k = k

    def __call__(self, X_: tensor, idxs=None, **f__kwargs):
        # Sample neighbors.
        end = self.idxs_start + self.k
        if end >= self.n:
            self.idxs_neighbors = randperm(self.n)
            self.idxs_start = 0
            end = self.k
        idxs_neighbors = self.idxs_neighbors[self.idxs_start:end]
        self.idxs_start = end

        # Build minibatch.
        if idxs is None:
            miniX = self.X
            miniX_ = X_
            miniw = self.w
            idxs = self.seq
        else:
            miniX = self.X[idxs]
            miniX_ = X_[idxs]
            # Ensure first weight is present to inhibit weight normalization inside `measure`.
            miniw = self.w if self.w is None else cat([self.w[:1], self.w[idxs_neighbors]])
        minin = len(idxs)

        # Calulate distances removing diagonal.
        # a = (self.seq != idxs[:, None])[:, idxs_neighbors]
        # print(self.X.shape, idxs_neighbors.shape, a.shape, torch.cdist(miniX, self.X[idxs_neighbors]).shape)
        miniD = torch.cdist(miniX, self.X[idxs_neighbors]).reshape(minin, -1)
        miniD_ = torch.cdist(miniX_, X_[idxs_neighbors]).reshape(minin, -1)
        # miniD = torch.cdist(miniX, self.X[idxs_neighbors])[a].reshape(self.k, -1)
        # miniD_ = torch.cdist(miniX_, X_[idxs_neighbors])[a].reshape(self.k, -1)
        # b = torch.arange(miniD_.size(0)).unsqueeze(1)

        miniD, miniD_ = cat([tensor([[0.0]] * minin), miniD], dim=1), cat([tensor([[0.0]] * minin), miniD_], dim=1)
        c = s = 0
        for minid, minid_ in zip(miniD, miniD_):
            s += self.measure(minid, minid_, miniw)
            c += 1
        return s / c
