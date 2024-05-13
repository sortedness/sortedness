#  10 lines of this file is based on the implementation at https://github.com/AlanBlanchet/tsne-pytorch
#
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
from numpy import ndarray
from torch import tensor, from_numpy, mean, mm
from torch.linalg import lstsq

from sortedness.new.quality.abs import Quality


class Linearity(Quality):
    # noinspection PyMissingConstructor
    def __init__(self, X: tensor):
        """

        >>> from torch import tensor
        >>> import torch
        >>> torch.manual_seed(0)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
        <torch._C.Generator object at 0x...>
        >>> X = torch.randn(100, 3)
        >>> X_ = torch.randn(100, 2)
        >>> round(float(Linearity(X)(X)), 8 )
        0.0
        >>> Linearity(X)(X_)
        tensor(0.8347)

        :param X: Original data.
        :param d0: First target dimensionality achieved through PCA.
        :param device: "cpu" or "cuda"
        :return:
        """
        if isinstance(X, ndarray):
            X = from_numpy(X.astype(np.float32))
        self.n = X.shape[0]
        self.X = X

    def __call__(self, X_: tensor, idxs=None, **f__kwargs):
        X = self.X
        if idxs is not None:
            X = X[idxs]
            X_ = X_[idxs]
        Vt = lstsq(X_, X).solution
        Xreconstructed = mm(X_, Vt)
        return 1 - mean((X - Xreconstructed) ** 2)

# torch.autograd.set_detect_anomaly(True)
