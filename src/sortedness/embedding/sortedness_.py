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
import math

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import halfnorm, rankdata
from torch import from_numpy, tensor, topk
from torch.optim import RMSprop
from torch.utils.data import Dataset, DataLoader

from sortedness.embedding.sigmas_ import findsigma, findweight
from sortedness.embedding.surrogate import cau, loss_function
from sortedness.local import remove_diagonal

activation_function_dct = {"tanh": torch.nn.Tanh, "sigm": torch.nn.Sigmoid, "relu": torch.nn.ReLU, "relu6": torch.nn.ReLU6}
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.00000001)


class Dt(Dataset):
    def __init__(self, X):
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def balanced_embedding_tacito(X, d=2, gamma=4,
                              # alpha=0.5,
                              beta=0.5, lambd=1,
                              neurons=30, epochs=100,
                              # batch_size=20,
                              # embedding_optimizer=RMSprop,
                              # min_K=100, max_K=1000,
                              seed=0,
                              # track_best_model=True,
                              # return_only_X_=True,
                              gpu=False, verbose=False, **embedding_optimizer__kwargs):
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
    beta
        Parameter to balance between local and global. 0 is totally local. 1 is totally global.
    lambd
        Regularizer. Surrogate function tends to (non differentiable) Kendall tau when lambd tends to 0.
    neurons
    epochs

    seed
        int

    return_only_X_
        Return `X_` or `(X_, model, quality)`?
    gpu
        Whether to use GPU.
    verbose
        Print information about best epoch?
            best_epoch, best_quality
    embedding_optimizer__kwargs
        Arguments for `learner`. Intended to expose for tunning the hyperparameters that affect speed or quality of learning.
        Default arguments for RMSprop:
            lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False

    Returns
    -------
    Transformed `d`-dimensional data as a numpy float array.

    """

    """
    k
        Number of nearest neighbors to consider for local optimization. This avoids useless sorting of neighbors with insignificant weights (as explained above for parameter `gamma`).
    K
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_K`.
    alpha
        Parameter to balance between continuity and trustworthiness. 0 is only continuity. 1 is only trustworthiness.
        default=0.5
            Consider neighborhood order on both X and X_ for weighting. Take the mean between extrusion and intrusion emphasis.

    batch_size
    embedding_optimizer
        Callable to perform gradient descent. See learner_parameters below.
        Default = RMSProp
    min_K
        Lower bound for the number of "neighbors" to sample when `K` is dynamic.
    max_K
        Upper bound for the number of "neighbors" to sample when `K` is dynamic.

    track_best_model
        Whether to return the best result (default) or the last one.
    """
    try:
        k = next(i for i, v in enumerate(cau(r, gamma) for r in range(len(X))) if v / cau(0, gamma) < 0.001)  # todo: ajeitar isso antes de empacotar
    except:
        k = len(X)
    # if verbose:
    #     print(f"Estimated k={k}")
    alpha = 1
    K = "sqrt"
    batch_size = 12
    embedding_optimizer = RMSprop
    min_K = 100
    max_K = 1000
    track_best_model = True
    return_only_X_ = False

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
    if "alpha_" in embedding_optimizer__kwargs:
        embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")

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
    best_quality_surrogate = best_epoch = -1
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

                quality, mu_local, mu_global, tau_local, tau_global = loss_function(miniD, miniD_, miniDsorted, miniidxs_by_D, k, K, w, alpha, beta, lambd, min_K, max_K)
                if track_best_model and quality > best_quality_surrogate:
                    best_quality_surrogate = quality
                    best_X_ = X_
                    best_epoch = i
                    best_dct = copy.deepcopy(model.state_dict())

                learning_optimizer.zero_grad()
                (-quality).backward()
                learning_optimizer.step()

    if track_best_model:
        model = M()
        model.load_state_dict(best_dct)
        X_ = best_X_
        if verbose:
            print(f"{best_epoch=} {float(best_quality_surrogate)=}", flush=True)

    X_ = X_.detach().cpu().numpy().astype(float)

    if return_only_X_:
        return X_

    return X_, model, float(best_quality_surrogate)


def balanced_embedding(X, d=2, kappa=5, K: int = "sqrt", alpha=0.5, beta=0.5, lambd=0.5,
                       hidden_layers=[50], epochs=100, batch_size=20, activation_functions=["relu"], embedding_optimizer=RMSprop,
                       min_K=100, max_K=1000, pct=90, epsilon=0.00001, seed=0, track_best_model=True, return_only_X_=True,
                       hyperoptimizer=None, sgd_alpha=0.01, sgd_mu=0.0, gpu=False, verbose=False, **embedding_optimizer__kwargs):
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
    K
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_K`.
    alpha
        Parameter to analogously balance between continuity and trustworthiness. 0 is only continuity. 1 is only trustworthiness.
        default=0.5
            Consider neighborhood order on both X and X_ for weighting. Take the mean between extrusion and intrusion emphasis.
    beta
        Parameter to balance between local and global. 0 is totally local. 1 is totally global.
    lambd
        Regularizer. Surrogate function tends to (non differentiable) Kendall tau when lambd tends to 0.
    neurons
    epochs
    batch_size
    embedding_optimizer
        Callable to perform gradient descent. See learner_parameters below.
        Default = RMSProp
    min_K
        Lower bound for the number of "neighbors" to sample when `K` is dynamic.
    max_K
        Upper bound for the number of "neighbors" to sample when `K` is dynamic.
    seed
        int
    track_best_model
        Whether to return the best result (default) or the last one.
    return_only_X_
        Return `X_` or `{"X_": X_, "model": model, "epoch": best_epoch, "surrogate_quality": float}`?
    gpu
        Whether to use GPU.
    verbose
        Print information about best epoch?
            best_epoch, best_quality_surrogate, best_dct
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
            layers = [torch.nn.Linear(X.shape[1], hidden_layers[0]), activation_function_dct[activation_functions[0]]()]
            previous = hidden_layers[0]
            for neurons, af in zip(hidden_layers[1:], activation_functions[1:]):
                layers.extend([torch.nn.Linear(previous, neurons), activation_function_dct[af]()])
                previous = neurons
            layers.append(torch.nn.Linear(previous, d))
            self.encoder = torch.nn.Sequential(*layers)
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
    if "alpha_" in embedding_optimizer__kwargs:
        embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")

    v = X.shape[0] - 1
    if K == "sqrt":
        K = max(min_K, min(max_K, int(math.sqrt(v))))
    if K > v:
        K = v
    if K < 1:
        raise Exception(f"`K` must be greater than 1: {K} > 1")
    if not (0 <= alpha <= 1):
        raise Exception(f"`alpha` outside valid range: 0 <= {alpha} <= 1")
    if not (0 <= beta <= 1):
        raise Exception(f"`beta` outside valid range: 0 <= {beta} <= 1")
    if not (0.00001 <= lambd <= 100):
        raise Exception(f"`lambd` outside valid range: 0.0001 <= {lambd} <= 100")

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

    sigma = findsigma(pct, kappa)
    k = int(halfnorm.ppf(1 - epsilon, 0, sigma))
    w = tensor([findweight(x, sigma) for x in range(k)])

    Dsorted, idxs_by_D = (None, None) if alpha == 1 else topk(D, k, largest=False, dim=1)

    if hyperoptimizer is None:
        if "alpha_" in embedding_optimizer__kwargs:
            embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")
        learning_optimizer = embedding_optimizer(model.parameters(), **embedding_optimizer__kwargs)
    else:
        from gradient_descent_the_ultimate_optimizer import gdtuo
        if hyperoptimizer == 1:
            optim = gdtuo.RMSProp(optimizer=gdtuo.SGD(alpha=sgd_alpha, mu=sgd_mu))
        elif hyperoptimizer == 2:
            optim = gdtuo.Adam(optimizer=gdtuo.SGD(alpha=sgd_alpha, mu=sgd_mu))
        else:
            raise Exception(f"Invalid optim: {hyperoptimizer}")
        mw = gdtuo.ModuleWrapper(model, optimizer=optim)
        mw.initialize()

    model.train()
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    best_quality_surrogate = best_epoch = -2
    with ((torch.enable_grad())):
        for i in range(epochs):
            for idx in loader:
                if hyperoptimizer is not None:
                    mw.begin()

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

                quality, mu_local, mu_global, tau_local, tau_global = loss_function(miniD, miniD_, miniDsorted, miniidxs_by_D, k, K, w, alpha, beta, lambd)
                if track_best_model and quality > best_quality_surrogate:
                    best_quality_surrogate = quality
                    best_X_ = X_
                    best_epoch = i
                    best_dct = copy.deepcopy(model.state_dict())

                if hyperoptimizer is None:
                    learning_optimizer.zero_grad()
                    (-quality).backward()
                    learning_optimizer.step()
                else:
                    mw.zero_grad()
                    loss = -quality
                    loss.backward(create_graph=True)
                    mw.step()

    if track_best_model:
        model = M()
        model.load_state_dict(best_dct)
        X_ = best_X_
        if verbose:
            print(f"{best_epoch=} {float(best_quality_surrogate)=}", flush=True)

    X_ = X_.detach().cpu().numpy().astype(float)

    if return_only_X_:
        return X_

    return {"X_": X_, "model": model, "epoch": best_epoch, "surrogate_quality": float(best_quality_surrogate)}


def step(X, idx, model, gpu, k, K, w, alpha, beta, lambd, learning_optimizer, mw=False):
    n = X.shape[0]
    if mw:
        learning_optimizer.begin()

    X_ = model(X)
    miniX_ = X_[idx]

    # Distance matrix without diagonal.
    l = len(idx)

    if 0:
        # magic (axis in isolation)
        quality = 0
        for col in range(X.shape[1]):
            Xax = X[:, col:col + 1]
            Dax = remove_diagonal(cdist(Xax, Xax))
            mx = np.max(Dax, axis=1, keepdims=True)
            Dax /= mx
            Dax = from_numpy(Dax).cuda() if gpu else from_numpy(Dax)
            miniDax = Dax[idx]
            if alpha == 1:
                miniDaxsorted = miniidxs_by_Dax = None
            else:
                miniDaxsorted, miniidxs_by_Dax = topk(miniDax, k, largest=False, dim=1)
            quality_, mu_local, mu_global, tau_local, tau_global = loss_function(miniDax, miniD_, miniDaxsorted, miniidxs_by_Dax, k, K, w, alpha, beta, lambd)
            quality += quality_
        quality = quality / X.shape[1]
    elif 0:
        # local magic (manhattan over ranks before axis isolation)
        miniRaxes = []
        for col in range(X.shape[1]):
            Xax = X[:, col:col + 1]
            Dax = remove_diagonal(cdist(Xax, Xax))
            mx = np.max(Dax, axis=1, keepdims=True)
            Dax /= mx
            Dax = from_numpy(Dax).cuda() if gpu else from_numpy(Dax)
            miniDax = Dax[idx]
            r = rankdata(miniDax, axis=1, method="average")
            miniRaxes.append(r)
        miniidxs_by_D = None if alpha == 1 else np.argsort(np.sum(np.stack(miniRaxes), axis=0), axis=1)[:, :k]
        quality = 0

        for col in range(X.shape[1]):
            Xax = X[:, col:col + 1]
            Dax = remove_diagonal(cdist(Xax, Xax))
            mx = np.max(Dax, axis=1, keepdims=True)
            Dax /= mx
            Dax = from_numpy(Dax).cuda() if gpu else from_numpy(Dax)
            miniDax = Dax[idx]
            miniDaxsorted = None if alpha == 1 else np.take_along_axis(miniDax, miniidxs_by_D, axis=1)
            Xax_ = X_[:, col:col + 1]
            miniXax_ = miniX_[:, col:col + 1]
            miniD_ = torch.cdist(miniXax_, Xax_)[torch.arange(n) != idx[:, None]].reshape(l, -1)
            quality_, mu_local, mu_global, tau_local, tau_global = loss_function(miniDax, miniD_, miniDaxsorted, miniidxs_by_D, k, K, w, alpha, beta, lambd)
            quality += quality_
        quality = quality / X.shape[1]
    else:
        # local magic (relative-density)
        miniRaxes = []
        for col in range(X.shape[1]):
            Xax = X[:, col:col + 1]
            Dax = remove_diagonal(cdist(Xax, Xax))
            mx = np.max(Dax, axis=1, keepdims=True)
            Dax /= mx
            Dax = from_numpy(Dax).cuda() if gpu else from_numpy(Dax)
            miniDax = Dax[idx]
            r = rankdata(miniDax, axis=1, method="average")
            miniRaxes.append(r)
        miniidxs_by_D = None if alpha == 1 else np.argsort(np.prod(np.stack(miniRaxes), axis=0), axis=1)[:, :k]
        quality = 0

        for col in range(X.shape[1]):
            Xax = X[:, col:col + 1]
            Dax = remove_diagonal(cdist(Xax, Xax))
            mx = np.max(Dax, axis=1, keepdims=True)
            Dax /= mx
            Dax = from_numpy(Dax).cuda() if gpu else from_numpy(Dax)
            miniDax = Dax[idx]
            miniDaxsorted = None if alpha == 1 else np.take_along_axis(miniDax, miniidxs_by_D, axis=1)
            Xax_ = X_[:, col:col + 1]
            miniXax_ = miniX_[:, col:col + 1]
            miniD_ = torch.cdist(miniXax_, Xax_)[torch.arange(n) != idx[:, None]].reshape(l, -1)
            quality_, mu_local, mu_global, tau_local, tau_global = loss_function(miniDax, miniD_, miniDaxsorted, miniidxs_by_D, k, K, w, alpha, beta, lambd)
            quality += quality_
        quality = quality / X.shape[1]

    learning_optimizer.zero_grad()
    loss = -quality
    if mw:
        loss.backward(create_graph=True)
    else:
        loss.backward()
    learning_optimizer.step()
    return X_, quality


def balanced_embedding_(X, d=2, kappa=5, K: int = "sqrt", alpha=0.5, beta=0.5, lambd=0.5,
                        hidden_layers=[50], epochs=100, batch_size=20, activation_functions=["relu"], embedding_optimizer=RMSprop,
                        min_K=100, max_K=1000, pct=90, epsilon=0.00001, seed=0, track_best_model=True, return_only_X_=True,
                        hyperoptimizer=None, sgd_alpha=0.01, sgd_mu=0.0, gpu=False, verbose=False, **embedding_optimizer__kwargs):
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
    K
        int:    Number of "neighbors" to sample for global optimization.
        "sqrt": Take the square root of the number of points limited by `max_K`.
    alpha
        Parameter to analogously balance between continuity and trustworthiness. 0 is only continuity. 1 is only trustworthiness.
        default=0.5
            Consider neighborhood order on both X and X_ for weighting. Take the mean between extrusion and intrusion emphasis.
    beta
        Parameter to balance between local and global. 0 is totally local. 1 is totally global.
    lambd
        Regularizer. Surrogate function tends to (non differentiable) Kendall tau when lambd tends to 0.
    neurons
    epochs
    batch_size
    embedding_optimizer
        Callable to perform gradient descent. See learner_parameters below.
        Default = RMSProp
    min_K
        Lower bound for the number of "neighbors" to sample when `K` is dynamic.
    max_K
        Upper bound for the number of "neighbors" to sample when `K` is dynamic.
    seed
        int
    track_best_model
        Whether to return the best result (default) or the last one.
    return_only_X_
        Return `X_` or `(X_, model, quality_surrogate)`?
    gpu
        Whether to use GPU.
    verbose
        Print information about best epoch?
            best_epoch, best_quality_surrogate, best_dct
    embedding_optimizer__kwargs
        Arguments for `learner`. Intended to expose for tunning the hyperparameters that affect speed or quality of learning.
        Default arguments for RMSprop:
            lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False

    Returns
    -------
    Transformed `d`-dimensional data as a numpy float array.

    """
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    X = (X - mn) / (mx - mn)
    X = X[:, np.any(X, axis=0)]

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = [torch.nn.Linear(X.shape[1], hidden_layers[0]), activation_function_dct[activation_functions[0]]()]
            previous = hidden_layers[0]
            for neurons, af in zip(hidden_layers[1:], activation_functions[1:]):
                layers.extend([torch.nn.Linear(previous, neurons), activation_function_dct[af]()])
                previous = neurons
            layers.append(torch.nn.Linear(previous, d))
            self.encoder = torch.nn.Sequential(*layers)
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
    if "alpha_" in embedding_optimizer__kwargs:
        embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")

    v = X.shape[0] - 1
    if K == "sqrt":
        K = max(min_K, min(max_K, int(math.sqrt(v))))
    if K > v:
        K = v
    if K < 1:
        raise Exception(f"`K` must be greater than 1: {K} > 1")
    if not (0 <= alpha <= 1):
        raise Exception(f"`alpha` outside valid range: 0 <= {alpha} <= 1")
    if not (0 <= beta <= 1):
        raise Exception(f"`beta` outside valid range: 0 <= {beta} <= 1")
    if not (0.00001 <= lambd <= 100):
        raise Exception(f"`lambd` outside valid range: 0.0001 <= {lambd} <= 100")

    torch.manual_seed(seed)
    model = M()
    if gpu:
        model.cuda()

    X = X.astype(np.float32)
    n = X.shape[0]
    X = from_numpy(X).cuda() if gpu else from_numpy(X)

    sigma = findsigma(pct, kappa)
    k = int(halfnorm.ppf(1 - epsilon, 0, sigma))
    w = tensor([findweight(x, sigma) for x in range(k)])

    if hyperoptimizer is None:
        if "alpha_" in embedding_optimizer__kwargs:
            embedding_optimizer__kwargs["alpha"] = embedding_optimizer__kwargs.pop("alpha_")
        learning_optimizer = embedding_optimizer(model.parameters(), **embedding_optimizer__kwargs)
        mw = False
    else:
        from gradient_descent_the_ultimate_optimizer import gdtuo
        if hyperoptimizer == 1:
            optim = gdtuo.RMSProp(optimizer=gdtuo.SGD(alpha=sgd_alpha, mu=sgd_mu))
        elif hyperoptimizer == 2:
            optim = gdtuo.Adam(optimizer=gdtuo.SGD(alpha=sgd_alpha, mu=sgd_mu))
        else:
            raise Exception(f"Invalid optim: {hyperoptimizer}")
        learning_optimizer = gdtuo.ModuleWrapper(model, optimizer=optim)
        learning_optimizer.initialize()
        mw = True

    model.train()
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    best_quality_surrogate = best_epoch = -2
    with ((torch.enable_grad())):
        for i in range(epochs):
            for idx in loader:
                X_, quality = step(X, idx, model, gpu, k, K, w, alpha, beta, lambd, learning_optimizer, mw)
                if track_best_model and quality > best_quality_surrogate:
                    best_quality_surrogate = quality
                    best_X_ = X_
                    best_epoch = i
                    best_dct = copy.deepcopy(model.state_dict())

    if track_best_model:
        model = M()
        model.load_state_dict(best_dct)
        X_ = best_X_
        if verbose:
            print(f"{best_epoch=} {float(best_quality_surrogate)=}", flush=True)

    X_ = X_.detach().cpu().numpy().astype(float)

    if return_only_X_:
        return X_

    return {"X_": X_, "model": model, "epoch": best_epoch, "surrogate_quality": float(best_quality_surrogate)}
