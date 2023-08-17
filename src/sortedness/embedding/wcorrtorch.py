# Based on https://pypi.org/project/wcorr/

import numpy as np
import torch
from torch import cumsum, diff, hstack, tensor
from torchsort import soft_rank


def wcov(x, y, w, mx, my):
    return (w * (x - mx) * (y - my)).sum()


def pearson(x, y, w):
    """
    >>> pearson(np.array([1,2,3]), np.array([1,2,3]), np.array([1,2,3]))
    """
    ws = w.sum()
    mx = (x * w).sum() / ws
    my = (y * w).sum() / ws
    return wcov(x, y, w, mx, my) / (wcov(x, x, w, mx, mx) * wcov(y, y, w, my, my)).sqrt()


def spearman(x, y, w, regularization="l2", regularization_strength=1.):
    """
    >>> spearman(tensor([[1,2,3,4,5]]), tensor([[1,2,3,4,5]]), tensor([[1,2,3,4,5]]))
    """
    return pearson(wsrank(x, w, regularization=regularization, regularization_strength=regularization_strength),
                   wsrank(y, w, regularization=regularization, regularization_strength=regularization_strength), w)


def spearmangpt(x, y, w):
    """
    >>> spearmangpt(tensor([1, 2, 3, 4.]), tensor([1, 2, 4, 3.]), tensor([1, 1/2, 1/3, 1/4]))
    tensor(0.9629)
    >>> spearmangpt(tensor([1, 2, 3, 4.]), tensor([2, 1, 3, 4.]), tensor([1, 1/2, 1/3, 1/4]))
    tensor(0.4278)
    """
    return pearson(wrankgpt(x, w), wrankgpt(y, w), w)


def wrankgpt(x, w):
    """
    >>> wrankgpt(tensor([1, 2, 2, 3.]), tensor([1, 1, 1, 1.]))
    tensor([0.5000, 0.7500, 0.7500, 2.5000])
    >>> wrankgpt(tensor([1, 2, 3, 4.]), tensor([1, 1/2, 1/3, 1/4]))
    tensor([0.5000, 1.2500, 1.6667, 1.9583])
    """
    sorted_indices = torch.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_w = w[sorted_indices]

    unique_sorted_x, inverse_indices = torch.unique(sorted_x, return_inverse=True)
    counts = torch.bincount(inverse_indices)

    cumsum_w = torch.cumsum(sorted_w, dim=0) - sorted_w
    adjusted = (cumsum_w[inverse_indices] + sorted_w / 2) / counts[inverse_indices]

    adjusted_w_rank = torch.zeros_like(x, dtype=torch.float)
    adjusted_w_rank[sorted_indices] = adjusted

    return adjusted_w_rank


def wsrank(x, w, regularization="l2", regularization_strength=1.):
    """
    >>> import torch
    >>> soft_rank(torch.tensor([[1., 2, 2, 3]]))
    tensor([[1.5000, 2.5000, 2.5000, 3.5000]])
    >>> wsrank(torch.tensor([[1., 2, 2, 3]]), torch.tensor([1., 1, 1, 1]))
    tensor([1.5000, 2.5000, 2.5000, 3.5000])
    >>> wsrank(torch.tensor([[1., 2, 3, 4]]), torch.tensor([1., 1/2, 1/3, 1/4]))
    tensor([1.0000, 1.5000, 1.8333, 2.0833])
    >>> wsrank(torch.tensor([[1., 2, 3, 4]], requires_grad=True), torch.tensor([1., 1/2, 1/3, 1/4])).sum().backward()
    """
    r = soft_rank(x, regularization=regularization, regularization_strength=regularization_strength).view(x.shape[1])
    d = hstack([r[0], diff(r)])
    s = cumsum((d * w) / 1, dim=0)
    return s
