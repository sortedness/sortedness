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


def soft_merge(x0, x1, y0, y1, Sx, Sy, estimate="average", tau=True, lambd=1.0):
    """
    sorts on x

    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="lowest", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="average", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="highest", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0

    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.04726...
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    4.67889...
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.43661...

    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="lowest", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -13.0
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="average", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="highest", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0

    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="lowest", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -10.89364...
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="average", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -6.66168...
    >>> soft_merge([1,2,3], [1,3,5], [5,6,7], [8,9,10], (6, 9), (18, 27), estimate="highest", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -5.95750...

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
    if not x0:
        return x1, y1, 0
    if not x1:
        return x0, y0, 0
    # todo: test for all sorted
    if x0[0] <= x1[0]:
        l = len(x1)
        if estimate == "average":
            mx, my = Sx[1] / l, Sy[1] / l
        elif estimate == "lowest":
            mx, my = x1[0], y1[0]
        elif estimate == "highest":
            mx, my = x1[-1], y1[-1]
        else:
            raise Exception(f"Unknown: {estimate=}")
        tanx = tanh((mx - x0[0]) / lambd)
        tany = tanh((my - y0[0]) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        Sx, Sy = [Sx[0] - x0[0], Sx[1]], [Sy[0] - y0[0], Sy[1]]
        rx, ry, t = soft_merge(x0[1:], x1, y0[1:], y1, Sx, Sy, estimate, tau, lambd)
        x, y = x0[:1] + rx, y0[:1] + ry
    else:
        l = len(x0)
        if estimate == "average":
            mx, my = Sx[0] / l, Sy[0] / l
        elif estimate == "lowest":
            mx, my = x0[-1], y0[-1]
        elif estimate == "highest":
            mx, my = x0[0], y0[0]
        else:
            raise Exception(f"Unknown: {estimate=}")
        tanx = tanh((x1[0] - mx) / lambd)
        tany = tanh((y1[0] - my) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        Sx, Sy = [Sx[0], Sx[1] - x1[0]], [Sy[0], Sy[1] - y1[0]]
        rx, ry, t = soft_merge(x0, x1[1:], y0, y1[1:], Sx, Sy, estimate, tau, lambd)
        x, y = x1[:1] + rx, y1[:1] + ry
    return x, y, t + l * dt


def soft_sort(x, y, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.98512...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.82502...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.43067...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.55708...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.39414...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.066794947761027
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.15941...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.06679...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.848227...
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.56523...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.31617...
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...


    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -2.32587...
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...

    >>> soft_sort([1,2,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([2,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -4.0
    >>> soft_sort([1,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0
    >>> soft_sort([1,1,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> soft_sort([1,2,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0

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
    xl, yl = len(x), len(y)
    if xl == 1:
        return x, y, 0, x[0], y[0]
    mid = xl // 2
    Sx, Sy = [None, None], [None, None]
    sortedleftx, sortedlefty, t0, Sx[0], Sy[0] = soft_sort(x[:mid], y[:mid], estimate, tau, lambd)
    sortedrightx, sortedrighty, t1, Sx[1], Sy[1] = soft_sort(x[mid:], y[mid:], estimate, tau, lambd)
    mergedx, mergedy, t = soft_merge(sortedleftx, sortedrightx, sortedlefty, sortedrighty, Sx, Sy, estimate, tau, lambd)
    return mergedx, mergedy, t + t0 + t1, Sx[0] + Sx[1], Sy[0] + Sy[1]


def softtau(x, y, estimate="average", tau=True, lambd=1.0):
    """
    >>> a = list(range(1, 100))
    >>> b = list(reversed(a))
    >>> from random import shuffle
    >>> import random
    >>> random.seed(0)
    >>> softtau(a, a, estimate="highest", lambd=0.0001)
    1.0
    >>> softtau(a, b, estimate="highest", lambd=0.0001)
    -1.0
    >>> s = 0
    >>> for i in range(100):
    ...     shuffle(b)
    ...     s += softtau(a, b, estimate="highest", lambd=0.0001)
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.0007...

    :param x:
    :param y:
    :param estimate:
    :param tau:
    :param lambd:
    :return:
    """
    s = soft_sort(x, y, estimate=estimate, tau=tau, lambd=lambd)[2]
    n = len(x)
    return s / n / (n - 1) * 2
