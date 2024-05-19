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

from numpy import array


def wsoft_merge(x, y, w, leftidx, rightidx, Sx, Sy, W, estimate="average", tau=True, lambd=1.0):
    """
    sorts on x

    :param y:
    :param x:
    :param w:
    :param leftidx:     Indexer for `x`.
    :param rightidx:    Indexer for `y`.
    :param Sx:  (sum(left values of x), sum(right values of x))
    :param Sy:  (sum(left values of y), sum(right values of y))
    :param W:   (sum(left weights), sum(right weights))
    :param estimate: All options have the same computational cost in practice.
        * "lowest"  →   most conservative; adopt the first difference as if it was the value for all pairs.
        * "average" →   most comprehensive for optimization as all items are included in the calculation;
                        adopt the average difference as if it was the value for all pairs.
        * "highest" →   most optimistic; adopt the last difference as if it was the value for all pairs;
                        along with "lowest", useful to estimate bounds.
    :param tau:     Approximate Kendall tau, instead of focusing on taking into account the magnitude of agreements and disagreements.
        * True  →   More interpretable.
        * False →   Best for optimization, if distances are to be somewhow preserved and if ties are causing problems.
                    The higher the better: (-inf, 0]
    :param lambd:   Regularizer.
                    Surrogate function tends to (non differentiable) Kendall tau when `lambd` tends to 0.
    :return:
    """
    ll, lr = len(leftidx), len(rightidx)
    if ll == 0:
        return rightidx, 0
    if lr == 0:
        return leftidx, 0
    # todo: test for all sorted
    lheadx, rheadx = x[leftidx[0]], x[rightidx[0]]
    if lheadx <= rheadx:
        if estimate == "average":
            mx, my = Sx[1] / lr, Sy[1] / lr
        elif estimate == "lowest":
            mx, my = rheadx, y[rightidx[0]]
        elif estimate == "highest":
            mx, my = x[rightidx[-1]], y[rightidx[-1]]
        else:
            raise Exception(f"Unknown: {estimate=}")
        lheady = y[leftidx[0]]
        tanx = tanh((mx - lheadx) / lambd)
        tany = tanh((my - lheady) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        weight = (w[leftidx[0]] * lr + W[1]) / 2
        Sx, Sy = [Sx[0] - lheadx, Sx[1]], [Sy[0] - lheady, Sy[1]]
        W = W[0] - w[leftidx[0]], W[1]
        idx, t = wsoft_merge(x, y, w, leftidx[1:], rightidx, Sx, Sy, W, estimate, tau, lambd)
        idx = leftidx[:1] + idx
    else:
        if estimate == "average":
            mx, my = Sx[0] / ll, Sy[0] / ll
        elif estimate == "lowest":
            mx, my = x[leftidx[-1]], y[leftidx[-1]]
        elif estimate == "highest":
            mx, my = lheadx, y[leftidx[0]]
        else:
            raise Exception(f"Unknown: {estimate=}")
        rheady = y[rightidx[0]]
        tanx = tanh((rheadx - mx) / lambd)
        tany = tanh((rheady - my) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        weight = (w[rightidx[0]] * ll + W[0]) / 2
        Sx, Sy = [Sx[0], Sx[1] - rheadx], [Sy[0], Sy[1] - rheady]
        W = W[0], W[1] - w[rightidx[0]]
        idx, t = wsoft_merge(x, y, w, leftidx, rightidx[1:], Sx, Sy, W, estimate, tau, lambd)
        idx = rightidx[:1] + idx
    return idx, t + weight * dt


def wsoft_sort(x, y, w, idx=None, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> from numpy import array
    >>> a = list(reversed(range(0, 5)))
    >>> w = array(a) / sum(a)
    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort([2,1,3,4,5], [1,2,3,4,5], w, estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort([1,1,3,4,5], [1,2,3,4,5], w, estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w, estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.80742...
    >>> wsoft_sort([2,1,3,4,5], [1,2,3,4,5], w, estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.40075...
    >>> wsoft_sort([1,1,3,4,5], [1,2,3,4,5], w, estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.60696...
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.60952...
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w, estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.60696...

    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort([2,1,3,4,5], [1,2,3,4,5], w, estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort([1,1,3,4,5], [1,2,3,4,5], w, estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w, estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> wsoft_sort([2,1,3,4,5], [1,2,3,4,5], w, estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> wsoft_sort([1,1,3,4,5], [1,2,3,4,5], w, estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w, estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0

    >>> wsoft_sort([1,2,3,4,5], [1,2,3,4,5], w, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort([2,1,3,4,5], [1,2,3,4,5], w, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.4
    >>> wsoft_sort([1,1,3,4,5], [1,2,3,4,5], w, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> wsoft_sort([1,1,3,4,5], [1,1,3,4,5], w, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> wsoft_sort([1,2,3,4,5], [1,1,3,4,5], w*0+1, tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0

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
    if idx is None:
        idx = list(range(len(x)))
    l = len(idx)
    if l == 1:
        return idx, 0, x[idx[0]], y[idx[0]], w[idx[0]]
    mid = l // 2
    Sx, Sy, W = [None, None], [None, None], [None, None]
    lsortedidx, t0, Sx[0], Sy[0], W[0] = wsoft_sort(x, y, w, idx[:mid], estimate, tau, lambd)
    rsortedidx, t1, Sx[1], Sy[1], W[1] = wsoft_sort(x, y, w, idx[mid:], estimate, tau, lambd)
    merged_idx, t = wsoft_merge(x, y, w, lsortedidx, rsortedidx, Sx, Sy, W, estimate, tau, lambd)
    return merged_idx, t + t0 + t1, Sx[0] + Sx[1], Sy[0] + Sy[1], W[0] + W[1]


def wsoft_tau(x, y, w, estimate="average", tau=True, lambd=1.0, normalized=True):
    """
    >>> a = list(range(100))
    >>> b = list(reversed(a))
    >>> from random import shuffle
    >>> import random
    >>> random.seed(0)
    >>> from numpy import array
    >>> w = array(b) / sum(b)
    >>> wsoft_tau(a, a, w, estimate="highest", lambd=0.0001)
    1.0
    >>> wsoft_tau(b[:10], a[:10], w[:10], estimate="highest", lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> wsoft_tau(b[:10], a[:10], w[:10], estimate="highest", lambd=0.0001, normalized=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.19090...
    >>> wsoft_tau(b, a, w, estimate="highest", lambd=0.0001)
    -1.0
    >>> s = 0
    >>> for i in range(100):
    ...     shuffle(b)
    ...     s += wsoft_tau(b, a, w, estimate="highest", lambd=0.0001)
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.00075...
    >>> a = list(range(6))
    >>> b = list(reversed(a))
    >>> w = array(b) / sum(b)
    >>> b = a.copy()
    >>> b[0], b[1] = a[1], a[0]
    >>> wsoft_tau(b[:4], a[:4], w[:4], lambd=0.0001, normalized=False)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.53333...
    >>> wsoft_tau(b[:4], a[:4], w[:4], lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.57142...
    >>> wsoft_tau(b[:4], a[:4], w[:4]*0+1, lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
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
    n = len(x)
    if n == 0:
        return 1
    total_weight = sum(w) if normalized else 1
    idx, s, _, _, _ = wsoft_sort(x, y, w, estimate=estimate, tau=tau, lambd=lambd)
    return min(1., max(-1., s / (n - 1) * 2 / total_weight))
