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
import torch
from numpy import ndarray
from sklearn.decomposition import PCA
from torch import tensor, from_numpy

from sortedness.new.quality.abs import Quality


class KLConcordanceGaussianCauchy(Quality):
    # noinspection PyMissingConstructor
    def __init__(self, X: tensor, d0=50, perplexity=30.0, device="cpu", earlyexag=True):
        """

        >>> from torch import tensor
        >>> torch.manual_seed(0)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
        <torch._C.Generator object at 0x...>
        >>> X = torch.randn(100, 3)
        >>> X_ = torch.randn(100, 2)
        >>> KLConcordanceGaussianCauchy(X, earlyexag=False)(X)
        tensor(-0.4456)
        >>> KLConcordanceGaussianCauchy(X, earlyexag=False)(X_)
        tensor(-1.3923)

        :param X: Original data.
        :param d0: First target dimensionality achieved through PCA.
        :param device: "cpu" or "cuda"
        :return:
        """
        if d0 < X.shape[1] < X.shape[0]:
            X = PCA(d0).fit_transform(X) if isinstance(X, ndarray) else PCA(d0).fit_transform(X.detach().cpu().numpy())
        if isinstance(X, ndarray):
            X = from_numpy(X.astype(np.float32))
        self.n = X.shape[0]
        # P = _x2p_torch(X, 1e-5, perplexity, device=torch.device(device))
        P = x2p(X, perplexity)
        P = P + P.t()
        P = torch.max(P, torch.tensor(1e-21, device=P.device))
        P = P / torch.sum(P)
        if earlyexag:
            P = P * 4.
        self.earlyexag = earlyexag
        self.P = P
        self.i = 0

    def __call__(self, X_: tensor, idxs=None, **f__kwargs):
        P = self.P
        sumX_ = torch.sum(X_ * X_, dim=1)
        num = -2. * torch.mm(X_, X_.t())
        num = 1. / (1. + (num + sumX_).t() + sumX_)
        num.fill_diagonal_(0)
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor(1e-21, device=Q.device))
        if self.earlyexag and self.i == 100:
            self.P = self.P / 4.
        self.i += 1
        return -torch.sum(P * torch.log(P / Q))


# torch.autograd.set_detect_anomaly(True)


def x2p(X, perplexity):
    D = torch.mm(X, X.t())
    N = D.size(0)
    P = torch.zeros_like(D)
    for i in range(N):
        sigma_i = binary_search_sigma(D[i], perplexity)
        affinities = torch.exp(-D[i] ** 2 / (2 * sigma_i ** 2))
        affinities[i] = 0  # Set self-affinity to zero
        sum_affinities = torch.sum(affinities)
        P[i] = affinities / sum_affinities
    return (P + torch.transpose(P, 0, 1)) / (2 * N)


def binary_search_sigma(distances, perplexity, tol=1e-5, max_iter=1000):
    low = torch.tensor(1e-20, dtype=distances.dtype, device=distances.device)
    high = torch.tensor(1e20, dtype=distances.dtype, device=distances.device)
    sigma = torch.ones_like(distances)
    for _ in range(max_iter):
        p = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        sum_p = torch.sum(p)
        entropy = torch.sum(-p * torch.log2(p / sum_p))
        entropy_diff = entropy - torch.log2(torch.tensor(perplexity))
        if torch.abs(entropy_diff) < tol:
            break
        if entropy_diff > 0:
            high = sigma
            sigma = 0.5 * (sigma + low)
        else:
            low = sigma
            sigma = 2.0 * sigma if torch.isinf(high).any() else 0.5 * (sigma + high)
    return sigma
