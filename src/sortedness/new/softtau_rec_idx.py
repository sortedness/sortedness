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


def soft_merge(x, y, leftidx, rightidx, Sx, Sy, estimate="average", tau=True, lambd=1.0):
    """
    sorts on x

    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="lowest", lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="average", lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="highest", lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0

    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.04726...
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    4.67889...
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.43661...

    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="lowest", tau=False, lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -13.0
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="average", tau=False, lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="highest", tau=False, lambd=0.0000001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0

    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="lowest", tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -10.89364...
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="average", tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -6.66168...
    >>> soft_merge([1,2,3,1,3,5], [5,6,7,8,9,10], [0,1,2], [3,4,5], (6, 9), (18, 27), estimate="highest", tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -5.95750...

    :param rightidx:
    :param Sy:
    :param Sx:
    :param x0:  Left halt of `x`.
    :param x1:  Right halt of `x`.
    :param y0:  Left halt of `y`.
    :param y1:  Right halt of `y`.
    :param Sy0: Sum of `y0` values.
    :param Sy1: Sum of `y1` values.
    :param Sx0: Sum of `x0` values.
    :param Sx1: Sum of `x1` values.
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
    if not leftidx:
        return rightidx, 0
    if not rightidx:
        return leftidx, 0
    # todo: test for all sorted
    lheadx, rheadx = x[leftidx[0]], x[rightidx[0]]
    if lheadx <= rheadx:
        l = len(rightidx)
        if estimate == "average":
            mx, my = Sx[1] / l, Sy[1] / l
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
        Sx, Sy = [Sx[0] - lheadx, Sx[1]], [Sy[0] - lheady, Sy[1]]
        idx, t = soft_merge(x, y, leftidx[1:], rightidx, Sx, Sy, estimate, tau, lambd)
        idx = leftidx[:1] + idx
    else:
        l = len(leftidx)
        if estimate == "average":
            mx, my = Sx[0] / l, Sy[0] / l
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
        Sx, Sy = [Sx[0], Sx[1] - rheadx], [Sy[0], Sy[1] - rheady]
        idx, t = soft_merge(x, y, leftidx, rightidx[1:], Sx, Sy, estimate, tau, lambd)
        idx = rightidx[:1] + idx
    return idx, t + l * dt


def soft_sort(x, y, idx=None, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.98512...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.82502...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.43067...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.55708...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.39414...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.066794947761027
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.15941...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="average")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.06679...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.848227...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.56523...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.31617...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest")[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...


    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -2.32587...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], tau=False)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -4.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[1]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
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
        return idx, 0, x[idx[0]], y[idx[0]]
    mid = l // 2
    Sx, Sy = [None, None], [None, None]
    lsortedidx, t0, Sx[0], Sy[0] = soft_sort(x, y, idx[:mid], estimate, tau, lambd)
    rsortedidx, t1, Sx[1], Sy[1] = soft_sort(x, y, idx[mid:], estimate, tau, lambd)
    merged_idx, t = soft_merge(x, y, lsortedidx, rsortedidx, Sx, Sy, estimate, tau, lambd)
    return merged_idx, t + t0 + t1, Sx[0] + Sx[1], Sy[0] + Sy[1]


def soft_tau(x, y, estimate="average", tau=True, lambd=1.0):
    """
    >>> a = list(range(1, 100))
    >>> b = list(reversed(a))
    >>> from random import shuffle
    >>> import random
    >>> random.seed(0)
    >>> soft_tau(a, a, estimate="highest", lambd=0.0001)
    1.0
    >>> soft_tau(b, a, estimate="highest", lambd=0.0001)
    -1.0
    >>> s = 0
    >>> for i in range(100):
    ...     shuffle(b)
    ...     s += soft_tau(b, a, estimate="highest", lambd=0.0001)
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.00012...
    >>> a = list(range(6))
    >>> b = list(reversed(a))
    >>> w = array(b) / sum(b)
    >>> b = a.copy()
    >>> b[0], b[1] = a[1], a[0]
    >>> soft_tau(b[:4], a[:4], lambd=0.0001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.66666...

    :param x:
    :param y:
    :param estimate:
    :param tau:
    :param lambd:
    :return:
    """
    s = soft_sort(x, y, estimate=estimate, tau=tau, lambd=lambd)[1]
    n = len(x)
    return s / n / (n - 1) * 2
