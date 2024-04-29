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
from sklearn.base import TransformerMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin
from torch import from_numpy
from torch.nn import Module
from torch.optim import RMSprop

from sortedness.new.transformation import transform


class SKTransformer(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """
    sklearn-friendly transformation from one space to another, e.g., dimensionality reduction optimizing the given function

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
    >>> skt = SKTransformer(f=Calmness(X), return_only_X_=False)
    >>> skt.fit(X)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    SKTransformer(f=..., return_only_X_=False)
    >>> X_, best_epoch, best_torch_model, best_quality_surrogate = skt.last_result.values()
    >>> X.shape, X_.shape, skt.transform(X).shape
    (torch.Size([20, 64]), torch.Size([20, 2]), torch.Size([20, 2]))
    >>> best_epoch, best_quality_surrogate  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    (87, 0.60...)
    """

    def __init__(self, f: callable,
                 ann: Module | int = None, epochs=100, batch_size=20, ann_optimizer=RMSprop, ann_optimizer__kwargs=None,
                 hyperoptimizer=None,
                 seed=0, gpu=False, verbose=False, **f__kwargs):
        self.f = f
        self.ann = ann
        self.epochs = epochs
        self.batch_size = batch_size
        self.ann_optimizer = ann_optimizer
        self.ann_optimizer__kwargs = ann_optimizer__kwargs
        self.hyperoptimizer = hyperoptimizer
        self.seed = seed
        self.gpu = gpu
        self.verbose = verbose
        self.f__kwargs = f__kwargs
        self.last_result = None

    def fit(self, X, y=None, plot=False, plot_labels=None, plot_colors=None, label_size=12, marker_size=50):
        """
        :param X:
            Matrix with an instance per row (often a high-dimensional data matrix).
        :param y:
            Unused.
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
        :return:
            Self
        """
        _ = y
        if isinstance(X, ndarray):
            X = from_numpy(X.astype(np.float32))
        self.fit_transform(X, None, plot, plot_labels, plot_colors, label_size, marker_size)
        return self

    def transform(self, X):
        if isinstance(X, ndarray):
            asnumpy = True
            X = from_numpy(X.astype(np.float32))
        else:
            asnumpy = False
        if self.gpu:
            X = X.cuda()
        X_ = self.last_result["best_torch_model"](X)
        if asnumpy:
            X_ = X_.detach().cpu().numpy().astype(float)
        return X_

    def fit_transform(self, X, y=None, plot=False, plot_labels=None, plot_colors=None, label_size=12, marker_size=50):  # **fit_params):
        """
        :param X:
            Matrix with an instance per row (often a high-dimensional data matrix).
        :param y:
            Unused.
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
        :return:
            Self
        """
        _ = y
        dct = transform(X, self.f, self.ann, self.epochs, self.batch_size, self.ann_optimizer, self.ann_optimizer__kwargs, self.hyperoptimizer, self.seed, False, self.gpu, plot, plot_labels, plot_colors, label_size, marker_size, self.verbose, **self.f__kwargs)
        self.last_result = dct
        return dct["X_"]
