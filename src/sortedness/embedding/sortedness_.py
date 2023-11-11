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
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import from_numpy, tensor
from torch.optim import RMSprop
from torch.utils.data import Dataset, DataLoader

from sortedness.embedding.surrogate import cau, loss_function

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)


class Dt(Dataset):
    def __init__(self, X):
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def balanced_embedding(X, symmetric, d=2, gamma=4, k=17, global_k: int = "sqrt", beta=0.5, smooothness_tau=1,
                       neurons=30, epochs=100, batch_size=20, embedding_optimizer=RMSprop,
                       min_global_k=100, max_global_k=1000, seed=0, gpu=False, **embedding_optimizer__kwargs):
    """
    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import StandardScaler
    >>> from numpy import random, round
    >>> digits = datasets.load_digits()
    >>> X = digits.images.reshape((len(digits.images), -1))[:20]
    >>> rnd = random.default_rng(0)
    >>> rnd.shuffle(X)
    >>> X = StandardScaler().fit_transform(X)
    >>> X_ = balanced_embedding(X, False, epochs=2)
    >>> X_.shape
    (20, 2)


    Parameters
    ----------
    X
        Matrix with an instance per row in a given space (often high-dimensional data).
    symmetric
        True:   Take the mean between extrusion and intrusion emphasis.
                See sortedness() documentation for details.
        False:  Weight by original distances (extrusion emphasis), not the projected distances.
    d
        Target dimensionality.
    gamma
        Cauchy distribution parameter. Higher values increase the number of neighbors with relevant weight values.
    k
        Number of nearest neighbors to consider for local optimization. This avoids useless sorting of neighbors with insignificant weights (as explained above for parameter `gamma`).
    global_k
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_global_k`.
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
                torch.nn.Linear(X.shape[1], neurons), torch.nn.ReLU(),
                torch.nn.Linear(neurons, d)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(d, neurons), torch.nn.ReLU(),
                torch.nn.Linear(neurons, X.shape[1])
            )

        def forward(self, x):
            return self.encoder(x)

    torch.manual_seed(seed)
    if symmetric:
        print("warning: 'symmetric=True' not implemented")
    model = M()
    if gpu:
        model.cuda()

    X = X.astype(np.float32)
    n = X.shape[0]
    # R = from_numpy(rankdata(cdist(X, X), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(X, X), axis=1))
    D = cdist(X, X)
    D /= np.max(D, axis=1)
    Dtarget = from_numpy(D).cuda() if gpu else from_numpy(D)
    T = from_numpy(X).cuda() if gpu else from_numpy(X)
    w = cau(tensor(range(n)), gamma=gamma)

    learning_optimizer = embedding_optimizer(model.parameters(), **embedding_optimizer__kwargs)
    model.train()
    loader = DataLoader(Dt(T), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    with torch.enable_grad():
        for i in range(epochs):
            for idx in loader:
                encoded = model(T)
                expected_ranking_batch = Dtarget[idx]
                D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
                loss, mu_local, mu_global, tau_local, tau_global = loss_function(D_batch, expected_ranking_batch, k, global_k, w, beta, smooothness_tau, min_global_k, max_global_k)
                learning_optimizer.zero_grad()
                (-loss).backward()
                learning_optimizer.step()

    return model(T).detach().cpu().numpy().astype(float)
