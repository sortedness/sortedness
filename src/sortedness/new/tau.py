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
from copy import copy
from math import tanh


def merge(x0, x1, y0, y1):
    """
    sorts on x

    >>> merge([1,2,3], [1,3,5], [5,6,7], [8,9,10])
    ([1, 1, 2, 3, 3, 5], [5, 8, 6, 7, 9, 10], 5)
    >>> merge([1,2,3], [4,5,6], [1,2,3], [4,5,6])
    ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 9)

    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param t:
    :return:
    """
    if not x0:
        return x1, y1, 0
    if not x1:
        return x0, y0, 0
    if x0[0] <= x1[0]:
        rx, ry, t = merge(x0[1:], x1, y0[1:], y1)
        return x0[:1] + rx, y0[:1] + ry, t + len(x1)
    else:
        rx, ry, t = merge(x0, x1[1:], y0, y1[1:])
        return x1[:1] + rx, y1[:1] + ry, t - len(x0)


def sort(x, y):
    """y should be sorted

    >>> sort([1,2,3,4,5], [1,2,3,4,5])
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 10)
    >>> sort([2,1,3,4,5], [1,2,3,4,5])
    ([1, 2, 3, 4, 5], [2, 1, 3, 4, 5], 8)
    >>> sort([1,1,3,4,5], [1,2,3,4,5])
    ([1, 1, 3, 4, 5], [1, 2, 3, 4, 5], 10)
    >>> sort([1,1,3,4,5], [1,1,3,4,5])
    ([1, 1, 3, 4, 5], [1, 1, 3, 4, 5], 10)
    >>> sort([1,2,3,4,5], [1,1,3,4,5])
    ([1, 2, 3, 4, 5], [1, 1, 3, 4, 5], 10)

    >>> a = list(range(6))
    >>> b = a.copy()
    >>> b[0], b[1] = a[1], a[0]
    >>> sort(b[:4], a[:4])  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    ([0, 1, 2, 3], [1, 0, 2, 3], 4)
    """
    xl, yl = len(x), len(y)
    if xl == 1:
        return x, y, 0
    m = xl // 2
    # leftx, rightx = x[:m], x[m:]
    # lefty, righty = y[:m], y[m:]
    x0, y0, t0 = sort(x[:m], y[:m])
    x1, y1, t1 = sort(x[m:], y[m:])
    rx, ry, t = merge(x0, x1, y0, y1)
    return rx, ry, t + t0 + t1


def argmergesort(r, idxs=None):
    """
    >>> argmergesort([1,2,3])
    [0, 1, 2]
    >>> argmergesort([4,3,2,1,0])
    [4, 3, 2, 1, 0]
    >>> argmergesort([2,4,3,1,0,9])
    [4, 3, 0, 2, 1, 5]

    :param idxs:
    :param r:
    :return:
    """
    if idxs is None:
        idxs = list(range(len(r)))
    n = len(idxs)
    if n == 1:
        return idxs[:1]
    mid = n // 2
    left = argmergesort(r, idxs[:mid])
    right = argmergesort(r, idxs[mid:])
    i = j = k = 0
    while i < len(left) and j < len(right):
        if r[left[i]] <= r[right[j]]:
            idxs[k] = left[i]
            i += 1
        else:
            idxs[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        idxs[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        idxs[k] = right[j]
        j += 1
        k += 1
    return idxs


def tau(r):
    """
    >>> tau([1,2,3])
    1.0
    >>> tau([4,3,2,1,0])
    -1.0
    >>> tau([2,4,3,1,9,0])
    -0.2
    >>> tau([1,2,3,5,4])
    0.8
    >>> tau([1,2,3,3,4]), tau([1,2,3,4,4])
    (0.9, 0.9)
    >>> tau([5,4,4,3,2]), tau([5,5,4,3,2])
    (-0.9, -0.9)

    :param r:
    :return:
    """

    raise Exception(f" fix ties")

    def tau(r, idxs, start, end):
        if end - start == 1:
            return 0
        mid = (start + end) // 2
        a, b = tau(r, idxs, start, mid), tau(r, idxs, mid, end)
        agreements = a + b
        if r[idxs[mid - 1]] < r[idxs[mid]]:
            return agreements + (mid - start) * (end - mid)
        i = k = start
        j = mid
        while i < mid and j < end:
            diff = r[idxs[j]] - r[idxs[i]]
            if diff >= 0:
                agreements += end - j
                tmp[k] = idxs[i]
                i += 1
            elif diff < 0:
                agreements -= mid - i
                tmp[k] = idxs[j]
                j += 1
            # else:
            #     # TODO: fix ties
            #     # tmp[k] = idxs[i]
            #     # i += 1
            #     # k += 1
            #     tmp[k] = idxs[j]
            #     j += 1
            k += 1
        while i < mid:
            tmp[k] = idxs[i]
            i += 1
            k += 1
        while j < end:
            tmp[k] = idxs[j]
            j += 1
            k += 1
        k = start
        while k < end:
            idxs[k] = tmp[k]
            k += 1
        return agreements

    idxs = list(range(len(r)))
    tmp = copy(idxs)
    end = len(r)
    return tau(r, idxs, 0, end) / end / (end - 1) * 2


def softtau(y, x, lambd):
    """
    y should be sorted

    >>> from torch import tensor
    >>> import torch
    >>> from scipy.stats import kendalltau
    >>> from scipy.stats import weightedtau
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]))
    tensor(1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]))
    tensor(-1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([1,2,3,4,5]),  tensor([1,2,3,4,5]))
    tensor(1.)
    >>> softtau(tensor([1,2,3,4,5]), tensor([5,4,3,2,1]),  tensor([1,2,3,4,5]))
    tensor(-1.)
    >>> round(float(softtau(tensor([1,2,3,4,5]), tensor([1,3,2,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.8
    >>> round(float(softtau(tensor([1,2,3,4,5]), tensor([1,2,2,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.948683
    >>> round(float(softtau(tensor([1,2,2,4,5]), tensor([1,2,3,4,5]),  tensor([1,1,1,1,1]), .1)), 6)
    0.948683
    >>> round(kendalltau([1,2,2,4,5], [1,2,3,4,5])[0], 6)
    0.948683
    >>> round(kendalltau([1,2,3,4,5], [1,2,2,4,5])[0], 6)
    0.948683

    >>> cau = torch.tensor([0.07961783439490445, 0.07493443237167478, 0.06369426751592357, 0.05095541401273885, 0.03980891719745223, 0.031070374398011493, 0.02449779519843214, 0.019598236158745713, 0.01592356687898089, 0.013132838663077023, 0.010981770261366132, 0.00929843321400344, 0.007961783439490446, 0.006885866758478223, 0.006008893161879581])
    >>> # strong break of trustworthiness = the last distance value (the one with lower weight) becomes the nearest neighbor.
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0]), cau, 0.00001)), 5)
    0.83258
    >>> # order of importance is defined by weights, not by values of argument `a`.
    >>> #round(float(softtau(torch.tensor([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), torch.tensor([0,14,13,12,11,10,9,8,7,6,5,4,3,2,1]), cau, 0.00001)), 5) # strong break of trustworthiness
    >>> # weak break of trustworthiness = an intermediate distance value (one with intermediate weight) becomes the nearest neighbor.
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # a weaker break of trustworthiness
    0.88332
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # strong break of continuity
    0.53172
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,3,17,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # weaker break of continuity
    0.76555
    >>> round(float(softtau(torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]), torch.tensor([1,2,2,4,5,6,7,8,9,10,11,12,13,14,15]), cau, 0.00001)), 5) # weaker break of continuity
    0.98904

    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,4,5,6,7,8,9,10,11,12,13,14, 0], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.83258
    >>> #round(weightedtau([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], [0,14,13,12,11,10,9,8,7,6,5,4,3,2,1], rank=False, weigher=lambda r:cau[r])[0], 5)
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3, 0,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.88332
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [17, 2,3,4,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.53172
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,3,17,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.76555
    >>> round(weightedtau([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1,2,2,4,5,6,7,8,9,10,11,12,13,14,15], rank=False, weigher=lambda r:cau[r])[0], 5)
    0.98904
    """
    raise Exception(f"broken")
    def tau(x, y, idxs, start, end):
        if end - start == 1:
            return 0
        mid = (start + end) // 2
        a, b = tau(x, y, idxs, start, mid), tau(x, y, idxs, mid, end)
        agreements = a + b
        d = x[idxs[mid]] - x[idxs[mid - 1]]
        if d >= 0:
            squashed = tanh(d) * tanh(y[idxs[mid]] - y[idxs[mid - 1]])
            return agreements + (mid - start) * (end - mid) * squashed
        i = k = start
        j = mid
        while i < mid and j < end:
            d = x[idxs[j]] - x[idxs[i]]
            squashed = tanh(d) * tanh(y[idxs[j]] - y[idxs[i]])
            if d >= 0:
                agreements += (end - j) * squashed
                tmp[k] = idxs[i]
                i += 1
            else:
                agreements += (mid - i) * squashed
                tmp[k] = idxs[j]
                j += 1
            k += 1
        while i < mid:
            tmp[k] = idxs[i]
            i += 1
            k += 1
        while j < end:
            tmp[k] = idxs[j]
            j += 1
            k += 1
        k = start
        while k < end:
            idxs[k] = tmp[k]
            k += 1
        return agreements

    idxs = list(range(len(x)))
    tmp = copy(idxs)
    end = len(x)
    return tau(x, y, idxs, 0, end) / end / (end - 1) * 2
