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
from functools import partial

import numpy as np
import torch
from matplotlib import animation, pyplot as plt
from torch import tensor
from torch.nn import Module
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from sortedness.new.dt import Dt
from sortedness.new.m import M

activation_function_dct = {"tanh": torch.nn.Tanh, "sigm": torch.nn.Sigmoid, "relu": torch.nn.ReLU, "relu6": torch.nn.ReLU6}
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.000000001)


def step(ax, plot_labels, plot_colors, marker_size, verbose, loader, hyperoptimizer, mw, ann, X, f, f__kwargs, learning_optimizer, state, char_size, mod, i):
    X_: tensor = None
    quality = t = 0
    for idx in loader:
        t += 1
        if hyperoptimizer is not None:
            mw.begin()
        X_ = ann(X)
        miniloss = -f(X_, idx, **f__kwargs)

        if hyperoptimizer is None:
            learning_optimizer.zero_grad()
            miniloss.backward()
            learning_optimizer.step()
        else:
            mw.zero_grad()
            miniloss.backward(create_graph=True)
            mw.step()
        quality -= float(miniloss)
    quality /= t
    if quality > state["best_quality_surrogate"]:
        state["best_quality_surrogate"] = quality
        state["best_X_"] = X_
        state["best_epoch"] = i
        state["best_dct"] = copy.deepcopy(ann.state_dict())
    if i % mod == 0:
        if verbose:
            print(i, quality, state["best_quality_surrogate"])
        X_ = X_.detach().cpu().numpy()
        ax.cla()
        ax.scatter(X_[:, 0], X_[:, 1], s=marker_size, c=plot_colors, alpha=0.5)
        for j, l in enumerate(plot_labels):
            ax.text(X_[j, 0], X_[j, 1], l, size=char_size)
        plt.title(f"{i}:  {quality:.4f}  ", fontsize=16)
    return X_, quality


def transform(X, f: callable,
              ann: Module | int = None, epochs=100, batch_size=20, ann_optimizer=RMSprop, ann_optimizer__kwargs=None,
              hyperoptimizer=None,
              seed=0, return_only_X_=True, gpu=False, plot=False, plot_labels=None, plot_colors=None, label_size=12, marker_size=50, verbose=False, **f__kwargs):
    """ Transformation from one space to another, e.g., dimensionality reduction optimizing the given function.

    :param X:
        Matrix with an instance per row (often a high-dimensional data matrix).
    :param f:
        Target/surrogate quality function to maximize. Usually a callable class instantiated with `X`.
        Expected signature:
            *   `quality = f(X_: tensor, idxs, **f__kwargs)`
            *   X_: current transformed data
            *   idxs: indexes of current mini-batch
            *   return → float value indicating the quality of the transformation
    :param ann:
        A subclass of `torch.nn.Module`; a `dict` of kwargs for module `M`; or, an `int` value representing the number of dimensions of the output space.
        The module is an Artificial Neural Network which defines topology, activation functions, etc.
        The default (`ann=None`) means `ann=M(X, d=2, hidden_layers=[40, 20], activation_functions=["sigm", "relu"])`.
        An `int` as in `ann=3` means `ann=M(X, d=3, hidden_layers=[40, 20], activation_functions=["sigm", "relu"])`.
        A `dict` as in `ann=dct` where `dct={"d": 2, "hidden_layers": [6, 4, 3], "activation_functions": ["sigm", "sigm", "relu"]` means `ann=M(X, **dct)`.
    :param epochs:
        Number of full learning iterations. See `batch_size`.
    :param batch_size:
        Number of instances in each mini-batch.
        Each epoch iterates over all mini-batches.
    :param ann_optimizer:
        Callable to perform gradient descent. See `ann_optimizer__kwargs` below.
        Default = RMSProp.
    :param ann_optimizer__kwargs:
        Arguments for `ann_optimizer`. Intended to expose (for tunning) the hyperparameters that affect speed or quality of learning.
        Default arguments for RMSprop:
            lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, foreach=None, maximize=False, differentiable=False
    :param hyperoptimizer:
        If different from None, it will be used to perform gradient descent also on the hyperparameters.
        For instance, two possible hyperoptimizers are `gdtuo.RMSProp(optimizer=gdtuo.SGD(alpha=0.01, mu=0.0))` and
        `gdtuo.Adam(optimizer=gdtuo.SGD(alpha=0.01, mu=0.0))`.
    :param return_only_X_:
        Return only `X_` or the tuple `(X_, best_epoch, best_torch_model, best_quality_surrogate)`?
    :param seed:
        int
    :param gpu:
        Whether to use GPU.
    :param plot:
        Show progress visually.
    :param plot_labels:
        List of `str` to label each point.
    :param plot_colors:
        List of `str` to color each point.
    :param label_size:
        Font size of plot labels.
    :param marker_size:
        Size of plot markers.
    :param verbose:
        Print information about best epoch?
            best_epoch, best_quality_surrogate, best_dct
    :param f__kwargs:
        Any extra keyworded argument is passed to `f` along with `X_`,`idxs` at each epoch.
    :return:
        Transformed `d`-dimensional data as a numpy float array, if `return_only_X_=True`.
        `(X_, best_epoch, best_torch_model, best_quality_surrogate)`, otherwise.

    >>> from sklearn import datasets
    >>> from sklearn.preprocessing import StandardScaler
    >>> from numpy import random, round
    >>> import torch
    >>> pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    >>> digits = datasets.load_digits()
    >>> X = digits.images.reshape((len(digits.images), -1))[:20]
    >>> rnd = random.default_rng(0)
    >>> rnd.shuffle(X)
    >>> X = StandardScaler().fit_transform(X)
    >>> X = torch.from_numpy(X.astype(np.float32))
    >>> from sortedness.new.quality import Calmness
    >>> X_, best_epoch, best_torch_model, best_quality_surrogate = transform(X, f=Calmness(X), return_only_X_=False).values()
    >>> X.shape, X_.shape
    (torch.Size([20, 64]), torch.Size([20, 2]))
    >>> best_epoch, best_quality_surrogate  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    (87, 0.60...)
    """

    if ann_optimizer__kwargs is None:
        ann_optimizer__kwargs = {}
    if epochs < 1:
        raise Exception(f"`epochs` < 1")
    if batch_size < 1:
        raise Exception(f"`batch_size` < 1")

    torch.manual_seed(seed)
    if ann is None:
        ann = M(X, d=2, hidden_layers=[40, 20], activation_functions=["sigm", "relu"])
    elif isinstance(ann, int):
        ann = M(X, d=ann, hidden_layers=[40, 20], activation_functions=["sigm", "relu"])
    elif isinstance(ann, dict):
        ann = M(X, **ann)
    if gpu:
        ann.cuda()

    if hyperoptimizer is None:
        learning_optimizer = ann_optimizer(ann.parameters(), **ann_optimizer__kwargs)
        mw = None
    else:
        from gradient_descent_the_ultimate_optimizer import gdtuo
        mw = gdtuo.ModuleWrapper(ann, optimizer=hyperoptimizer)
        mw.initialize()
        learning_optimizer = None

    ann.train()
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, pin_memory=gpu)
    state = dict(best_quality_surrogate=-9999, best_X_=None, best_epoch=-9999, best_dct=None)
    with torch.enable_grad():
        if plot:
            fig, ax = plt.subplots(1, 1)
            ax.cla()
            mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            anim = animation.FuncAnimation(fig, partial(step, ax, plot_labels, plot_colors, marker_size, verbose, loader, hyperoptimizer, mw, ann, X, f, f__kwargs, learning_optimizer, state, label_size, 1))
            plt.show()
        else:
            for i in range(epochs):
                step(None, plot_labels, plot_colors, marker_size, verbose, loader, hyperoptimizer, mw, ann, X, f, f__kwargs, learning_optimizer, state, label_size, 1, i)
    best_torch_model = ann.clone()
    best_torch_model.load_state_dict(state["best_dct"])
    X_: tensor = state["best_X_"]
    if verbose:
        print(f"{state['best_epoch']=} {float(state['best_quality_surrogate'])=}", flush=True)

    if return_only_X_:
        return X_
    return {"X_": X_, "best_epoch": state["best_epoch"], "best_torch_model": best_torch_model, "best_quality_surrogate": float(state["best_quality_surrogate"])}
