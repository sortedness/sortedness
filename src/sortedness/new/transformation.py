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

import copy

import numpy as np
import torch
from torch import from_numpy
from torch.optim import RMSprop
from torch.utils.data import Dataset, DataLoader

activation_function_dct = {"tanh": torch.nn.Tanh, "sigm": torch.nn.Sigmoid, "relu": torch.nn.ReLU, "relu6": torch.nn.ReLU6}
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.000000001)


class Dt(Dataset):
    def __init__(self, X):
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def transform(X, f: callable, pre_f: callable = None, d=2,
              hidden_layers=[40, 20], epochs=100, batch_size=20, activation_functions=["sigm", "relu"],
              embedding_optimizer=RMSprop,
              seed=0, track_best_model=True, return_only_X_=True,
              hyperoptimizer=None, sgd_alpha=0.01, sgd_mu=0.0, gpu=False, verbose=False,
              **embedding_optimizer__kwargs):
    """
    Parameters
    ----------
    X
        Matrix with an instance per row (often high-dimensional data).
        asdasd
    f
        Target/surrogate function to maximize.
        Expected signature:
            *   `quality = f(X, X_, idx, **kwargs_f)`
            *   X: original data
            *   X_: current transformed data
            *   idx: indexes of current mini-batch
            *   kwargs_f: optional; it is the return value of `pre_f` when needed - see `pre_f` below.
            *   return → float value to maximize indicating the quality of the transformation
    pre_f
        Preliminar function to be called before first epoch.
        Intended for heavy calculations, e.g., distance matrix, that happen only once.
        Expected signature:
            *   `kwargs_f = f(X)`
            *   X: original data
            *   return → kwargs for `f`
    d
        Target dimensionality.
    hidden_layers
        Artificial neural network topology.
    epochs
        Number of full learning iterations. See `batch_size`.
    batch_size
        Number of instances in each mini-batch.
        Each epoch iterates over all mini-batches.

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

    torch.manual_seed(seed)
    model = M()
    if gpu:
        model.cuda()
    X = X.astype(np.float32)
    X = from_numpy(X).cuda() if gpu else from_numpy(X)
    if pre_f is not None:
        kwargs_f = pre_f(X)

    if hyperoptimizer is None:
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
                quality = f(X, X_, idx, **kwargs_f)
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
