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
from math import tanh

import numpy as np
from numpy import array


def merge(x, y, w, idx, sx, sy, sw, estimate, tau, lambd, tmp, start, mid, end):
    """
    sorts on x
    """
    # todo: criar opção de quasitau que lida com ties da forma que acho mais correta?
    # todo: add parameter to indicate the degree of approximation; from total ad hoc (1) to no approximation (0)
    # Example for a ranking with 8 numbers
    #   1/8 would mean the current setting:
    #       lowest → test at the start (0)
    #       average → test at the average (0→7)
    #       highest → test at end (7)
    #   2/8 would mean:
    #       lowest → test at the start (0) and second (1)
    #       average → test at the first half average (0→3) and second half average (4→7)
    #       highest → test at the before end (6) and end (7)
    #   4/8 would mean:
    #       lowest → test at 0,1,2,3
    #       average → test at averages 0→1,2→3,4→5,6→7
    #       highest → test at 4,5,6,7

    left_sx, right_sx = sx[start], sx[mid]
    left_sy, right_sy = sy[start], sy[mid]
    left_sw, right_sw = sw[start], sw[mid]
    if x[idx[mid - 1]] <= x[idx[mid]] and estimate is not None:
        ll = mid - start
        lr = end - mid
        if estimate == "average":
            lx, rx = left_sx / ll, right_sx / lr
            ly, ry = left_sy / ll, right_sy / lr
        elif estimate == "lowest":
            lx, rx = x[idx[mid - 1]], x[idx[mid]]
            ly, ry = y[idx[mid - 1]], y[idx[mid]]
        elif estimate == "highest":
            lx, rx = x[idx[end - 1]], x[idx[start]]
            ly, ry = y[idx[end - 1]], y[idx[start]]
        else:
            raise Exception(f"Unknown: {estimate=}")
        tanx = tanh((rx - lx) / lambd)
        tany = tanh((ry - ly) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        weight = (lr * left_sw + ll * right_sw) / 2
        return weight * dt

    k = i = start
    j = mid
    t = 0
    while i < mid and j < end:
        if x[idx[i]] <= x[idx[j]]:
            if estimate is None:
                tanx = np.tanh((x[idx[j:end]] - x[idx[i]]) / lambd)
                tany = np.tanh((y[idx[j:end]] - y[idx[i]]) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[i]] + w[idx[j:end]]) / 2
                t += np.sum(weight * dt)
            else:
                l = end - j
                if estimate == "average":
                    mx, my = right_sx / l, right_sy / l
                    left_sx -= x[idx[i]]
                    left_sy -= y[idx[i]]
                elif estimate == "lowest":
                    mx, my = x[idx[j]], y[idx[j]]
                elif estimate == "highest":
                    mx, my = x[idx[end - 1]], y[idx[end - 1]]
                else:
                    raise Exception(f"Unknown: {estimate=}")
                tanx = tanh((mx - x[idx[i]]) / lambd)
                tany = tanh((my - y[idx[i]]) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[i]] * l + right_sw) / 2
                t += weight * dt
                left_sw -= w[idx[i]]
            tmp[k] = idx[i]
            i += 1
        else:
            if estimate is None:
                tanx = np.tanh((x[idx[j]] - x[idx[i:mid]]) / lambd)
                tany = np.tanh((y[idx[j]] - y[idx[i:mid]]) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[j]] + w[idx[i:mid]]) / 2
                t += np.sum(weight * dt)
            else:
                l = mid - i
                if estimate == "average":
                    mx, my = left_sx / l, left_sy / l
                    right_sx -= x[idx[j]]
                    right_sy -= y[idx[j]]
                elif estimate == "lowest":
                    mx, my = x[idx[mid - 1]], y[idx[mid - 1]]
                elif estimate == "highest":
                    mx, my = x[idx[i]], y[idx[i]]
                else:
                    raise Exception(f"Unknown: {estimate=}")
                tanx = tanh((x[idx[j]] - mx) / lambd)
                tany = tanh((y[idx[j]] - my) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[j]] * l + left_sw) / 2
                t += weight * dt
                right_sw -= w[idx[j]]
            tmp[k] = idx[j]
            j += 1
        k += 1
    if i < mid:
        idx[k:end] = idx[i:mid]
    idx[start:k] = tmp[start:k]
    return t


# todo: tau=True could use the correct denominator, provavlmente tem que criar outro acumulador
#   den = sum(tana ** 2 * sw) * sum(tanb ** 2 * sw)

def wsoft_sort(x, y, w, idx, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> from numpy import array
    >>> a = list(reversed(range(0, 5)))
    >>> w = array(a, dtype=float) / sum(a)
    >>> wsoft_sort(array([1.0,2,3,4,5]), array([1.0,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest")  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.77947...
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest")  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.34044...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="highest")  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="highest")  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="highest")  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), tau=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), tau=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.4
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w*0+1, array([0,1,2,3,4]),  tau=False, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0

    >>> wsoft_sort(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort(array([2,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> wsoft_sort(array([1,1,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> wsoft_sort(array([1,2,3,4,5]), array([1,1,3,4,5]), w*0+1, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    :param w:
    :param idx:
    :param x:           Vector of numbers.
    :param y:           Ordered vector of numbers.
    :param estimate:    All options have the same computational cost in practice.
        * "lowest"  →   most conservative; adopt the first difference as if it was the value for all pairs.
        * "average" →   most comprehensive for optimization as all items are included in the calculation;
                        adopt the average difference as if it was the value for all pairs.
        * "highest" →   most optimistic; adopt the last difference as if it was the value for all pairs;
                        along with "lowest", useful to estimate bounds.
    :param tau:     Approximate Kendall tau, instead of focusing on taking into account the magnitude of agreements and disagreements.
        * True  →   More interpretable.
        * False →   Best for optimization, if distances are to be somewhow preserved and if ties are causing problems.
    :param lambd:   Regularizer.
                    Surrogate function tends to (non differentiable) Kendall tau when `lambd` tends to 0.
    :return:
    """
    # r = randperm(x.shape[0])
    # x = x[r]
    # y = y[r]
    # w = w[r]
    # idx = idx[r]
    n = len(idx)
    if n < 2:
        return 0
    tmp = idx.copy()
    sx, sy, sw = x.copy(), y.copy(), w.copy()
    t, wid = 0, 1
    while wid < n:
        wid2 = 2 * wid
        for start in range(0, n, wid2):
            mid = start + wid
            end = min(mid + wid, n)
            if mid < n:
                t_ = merge(x, y, w, idx, sx, sy, sw, estimate, tau, lambd, tmp, start, mid, end)
                t += t_
                sx[start] += sx[mid]
                sy[start] += sy[mid]
                sw[start] += sw[mid]
        wid *= 2
    return t


def wsoft_tau(x, y, w, idx, estimate="average", tau=True, lambd=1.0, normalized=True):
    """
    >>> from numpy import array
    >>> a = list(reversed(range(0, 5)))
    >>> w = array(a) / sum(a)
    >>> wsoft_tau(array([1,2,3,4,5]), array([1,2,3,4,5]), w, array([0,1,2,3,4]), lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.00000...
    >>> a = list(range(100))
    >>> b = list(reversed(a))
    >>> from random import shuffle
    >>> import random
    >>> random.seed(0)
    >>> from numpy import array
    >>> w = array(b) / sum(b)
    >>> idx = a.copy()
    >>> wsoft_tau(a, a, w, idx.copy(), estimate="highest", lambd=0.0001)
    1.0
    >>> wsoft_tau(b[:10], a[:10], w[:10], idx.copy()[:10], estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> wsoft_tau(b[:10], a[:10], w[:10], idx.copy()[:10], estimate="average", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> wsoft_tau(b[:10], a[:10], w[:10], idx.copy()[:10], estimate="lowest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> wsoft_tau(array(a[:10]), array(a[:10]), w[:10], idx.copy()[:10], estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.99999...
    >>> wsoft_tau(array(b[:10]), array(a[:10]), w[:10], idx.copy()[:10], estimate=None, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> wsoft_tau(b[:10], a[:10], w[:10], idx.copy()[:10], estimate="highest", lambd=0.0001, normalized=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.19090...
    >>> wsoft_tau(b, a, w, idx.copy(), estimate="highest", lambd=0.0001)
    -1.0
    >>> s = 0
    >>> for i in range(100):
    ...     shuffle(b)
    ...     s += wsoft_tau(b, a, w, idx.copy(), estimate="highest", lambd=0.0001)
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.00075...
    >>> random.seed(1000)
    >>> s = 0
    >>> c = 0
    >>> from sortedness.new.quality.measure.pairwise import softtau
    >>> from torch import from_numpy
    >>> for i in range(100):
    ...     shuffle(b)
    ...     r1 = wsoft_tau(array(b), array(a), w, idx.copy(), estimate=None)
    ...     r2 = softtau(from_numpy(array(b)), from_numpy(array(a)), from_numpy(w))
    ...     if abs(r1 - r2) < 0.01:
    ...         c += 1
    ...     s += r1
    >>> c / 100
    1.0
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.0005...
    >>> a = list(range(6))
    >>> b = list(reversed(a))
    >>> w = array(b) / sum(b)
    >>> b = a.copy()
    >>> b[0], b[1] = a[1], a[0]
    >>> wsoft_tau(b[:4], a[:4], w[:4], idx.copy()[:4], lambd=0.0001, normalized=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.53333...
    >>> wsoft_tau(b[:4], a[:4], w[:4], idx.copy()[:4], lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.57142...
    >>> wsoft_tau(b[:4], a[:4], w[:4]*0+1, idx.copy()[:4], lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.66666...

    :param x:
    :param y:
    :param estimate:
    :param tau:
    :param lambd:
    :param normalized:  Reweight `w` so that `sum(w) == 1`.
                    *   True:   tau ranges within [-1; 1].
                    *   False:  tau will not reach the extremes of [-1; 1] when `sum(w) < 1` even for a perfect correlation.
                                Consider absent weights like ties.
    :return:
    """
    total_weight = sum(w) if normalized else 1
    s = wsoft_sort(x, y, w, idx, estimate=estimate, tau=tau, lambd=lambd)
    n = len(x)
    # return min(1., max(-1., s / (n - 1) * 2 / total_weight))
    return s / (n - 1) * 2 / total_weight


"""
In [10]: from sortedness.new.wsofttau_ite_idx import wsoft_tau
    ...: from sortedness.new.quality.measure.pairwise import softtau
    ...: 
    ...: import torch
    ...: from numpy import array
    ...: lst = list(range(10000))
    ...: x = array(lst)
    ...: y = array(lst)
    ...: # y = torch.tensor([10,9,8,7,6,5.0, 4.0, 3.0, 2.0, 1.0])
    ...: w = array(list(reversed(lst)))
    ...: idx = array(lst)
    ...: tx = torch.tensor(lst)
    ...: ty = torch.tensor(lst)
    ...: #ty = torch.tensor([10,9,8,7,6,5.0, 4.0, 3.0, 2.0, 1.0])
    ...: tw = torch.tensor(list(reversed(lst)))
    ...: tidx = torch.tensor(lst)
    ...: 
    ...: %timeit softtau(tx, ty, tw, 0.0001)
    ...: %timeit wsoft_tau(x, y, w, idx, "average", True, 0.0001, True)
1.51 s ± 15.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
27.8 ms ± 326 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
"""

"""
In [5]: import torch
   ...: import rustil
   ...: 
   ...: lst = list(range(10000))
   ...: x = torch.tensor(lst)
   ...: y = torch.tensor(lst)
   ...: # y = torch.tensor([10,9,8,7,6,5.0, 4.0, 3.0, 2.0, 1.0])
   ...: w = torch.tensor(list(reversed(lst)))
   ...: 
   ...: %timeit rustil.wsoft_tau(x, y, w, "average", True, 0.0001, True)
482 ms ± 1.99 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
"""
