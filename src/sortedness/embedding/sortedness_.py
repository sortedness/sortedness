#  Copyright (c) 2023. Davi Pereira dos Santos
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
import copy

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import from_numpy, tensor, topk
from torch.optim import RMSprop
from torch.utils.data import Dataset, DataLoader

from sortedness.embedding.surrogate import cau, loss_function
from sortedness.local import remove_diagonal

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.000000001)


class Dt(Dataset):
    def __init__(self, X):
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def balanced_embedding(X, d=2, gamma=4, k=17, global_k: int = "sqrt", alpha=0.5, beta=0.5, smooothness_tau=1,
                       neurons=30, epochs=100, batch_size=20, embedding_optimizer=RMSprop,
                       min_global_k=100, max_global_k=1000, seed=0, track_best_model=True, return_only_X_=True,
                       gpu=False, **embedding_optimizer__kwargs):
    """
    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import StandardScaler
    >>> from numpy import random, round
    >>> digits = datasets.load_digits()
    >>> X = digits.images.reshape((len(digits.images), -1))[:20]
    >>> rnd = random.default_rng(0)
    >>> rnd.shuffle(X)
    >>> X = StandardScaler().fit_transform(X)
    >>> X_ = balanced_embedding(X, alpha=0, epochs=2)
    >>> X_.shape
    (20, 2)


    Parameters
    ----------
    X
        Matrix with an instance per row in a given space (often high-dimensional data).
    d
        Target dimensionality.
    gamma
        Cauchy distribution parameter. Higher values increase the number of neighbors with relevant weight values.
    k
        Number of nearest neighbors to consider for local optimization. This avoids useless sorting of neighbors with insignificant weights (as explained above for parameter `gamma`).
    global_k
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_global_k`.
    alpha
        Parameter to balance between continuity and trustworthiness. 0 is only continuity. 1 is only trustworthiness.
        default=0.5
            Consider neighborhood order on both X and X_ for weighting. Take the mean between extrusion and intrusion emphasis.
    beta
        Parameter to balance between local and global. 0 is totally local. 1 is totally global.
    smooothness_tau
        Regularizer. Surrogate function tends to (non differentiable) Kendall tau when smooothness_tau tends to 0.
    neurons
    epochs
    batch_size
    embedding_optimizer
        Callable to perform gradient descent. See learner_parameters below.
        Default = RMSProp
    min_global_k
        Lower bound for the number of "neighbors" to sample when `global_k` is dynamic.
    max_global_k
        Upper bound for the number of "neighbors" to sample when `global_k` is dynamic.
    seed
        int
    track_best_model
        Whether to return the best result (default) or the last one.
    return_only_X_
        Return `X_` or `(X_, model, quality)`?
    gpu
        Whether to use GPU.
    embedding_optimizer__kwargs
        Arguments for `learner`. Intended to expose for tunning the hyperparameters that affect speed or quality of learning.
        Default arguments for RMSprop: 
            lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False
            
    Returns
    -------
    Transformed `d`-dimensional data as a numpy float array.

    """

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], neurons), torch.nn.Tanh(),
                torch.nn.Linear(neurons, d)
            )
            # self.decoder = torch.nn.Sequential(
            #     torch.nn.Linear(d, neurons), torch.nn.ReLU(),
            #     torch.nn.Linear(neurons, X.shape[1])
            # )

        def forward(self, x):
            return self.encoder(x)

    if epochs < 1:
        raise Exception(f"`epochs` < 1")
    if batch_size < 1:
        raise Exception(f"`batch_size` < 1")
    torch.manual_seed(seed)
    model = M()
    if gpu:
        model.cuda()

    X = X.astype(np.float32)
    n = X.shape[0]

    D = remove_diagonal(cdist(X, X))  # todo: distance by batches in a more memory-friendly way
    D /= np.max(D, axis=1, keepdims=True)

    X = from_numpy(X).cuda() if gpu else from_numpy(X)
    D = from_numpy(D).cuda() if gpu else from_numpy(D)

    Dsorted, idxs_by_D = (None, None) if alpha == 1 else topk(D, k, largest=False, dim=1)
    w = cau(tensor(range(n)), gamma=gamma)

    if "alpha_" in embedding_optimizer__kwargs:
        embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")
    learning_optimizer = embedding_optimizer(model.parameters(), **embedding_optimizer__kwargs)
    model.train()
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    best_quality = -1
    with ((torch.enable_grad())):
        for i in range(epochs):
            for idx in loader:
                X_ = model(X)
                miniX_ = X_[idx]
                miniD = D[idx]
                if alpha == 1:
                    miniDsorted = None
                    miniidxs_by_D = None
                else:
                    miniDsorted = Dsorted[idx]
                    miniidxs_by_D = idxs_by_D[idx]

                # Distance matrix without diagonal.
                l = len(idx)
                miniD_ = torch.cdist(miniX_, X_)[torch.arange(n) != idx[:, None]].reshape(l, -1)

                quality, mu_local, mu_global, tau_local, tau_global = loss_function(miniD, miniD_, miniDsorted, miniidxs_by_D, k, global_k, w, alpha, beta, smooothness_tau, min_global_k, max_global_k)
                if track_best_model and quality > best_quality:
                    best_quality = quality
                    best_X_ = X_
                    best_dct = copy.deepcopy(model.state_dict())

                learning_optimizer.zero_grad()
                (-quality).backward()
                learning_optimizer.step()

    if track_best_model:
        model = M()
        model.load_state_dict(best_dct)
        X_ = best_X_

    X_ = X_.detach().cpu().numpy().astype(float)

    if return_only_X_:
        return X_

    return X_, model, best_quality
