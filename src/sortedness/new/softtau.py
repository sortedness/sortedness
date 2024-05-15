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


def softmerge(x0, x1, y0, y1, Sx0, Sx1, Sy0, Sy1, estimate="average", tau=True, lambd=1.0):
    """
    sorts on x

    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="lowest", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="average", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="highest", lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.0

    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.04726...
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    4.67889...
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.43661...

    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="lowest", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -13.0
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="average", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="highest", tau=False, lambd=0.0000001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -8.0

    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="lowest", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -10.89364...
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="average", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -6.66168...
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 9, 18, 27, estimate="highest", tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -5.95750...

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
            mx, my = Sx1 / l, Sy1 / l
        elif estimate == "lowest":
            mx, my = x1[0], y1[0]
        elif estimate == "highest":
            mx, my = x1[-1], y1[-1]
        else:
            raise Exception(f"Unknown: {estimate=}")
        tanx = tanh((mx - x0[0]) / lambd)
        tany = tanh((my - y0[0]) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        rx, ry, t = softmerge(x0[1:], x1, y0[1:], y1, Sx0 - x0[0], Sx1, Sy0 - y0[0], Sy1, estimate, tau, lambd)
        x, y = x0[:1] + rx, y0[:1] + ry
    else:
        l = len(x0)
        if estimate == "average":
            mx, my = Sx0 / l, Sy0 / l
        elif estimate == "lowest":
            mx, my = x0[-1], y0[-1]
        elif estimate == "highest":
            mx, my = x0[0], y0[0]
        else:
            raise Exception(f"Unknown: {estimate=}")
        tanx = tanh((x1[0] - mx) / lambd)
        tany = tanh((y1[0] - my) / lambd)
        dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
        rx, ry, t = softmerge(x0, x1[1:], y0, y1[1:], Sx0, Sx1 - x1[0], Sy0, Sy1 - y1[0], estimate, tau, lambd)
        x, y = x1[:1] + rx, y1[:1] + ry
    return x, y, t + l * dt


def sort(x, y, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.98512...
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.82502...
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.43067...
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="highest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.41786...

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="average", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.55708...
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.39414...
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.066794947761027
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.15941...
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="average")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.06679...

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    10.0
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    8.0
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest", lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    9.0

    >>> sort([1,2,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.848227...
    >>> sort([2,1,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    5.56523...
    >>> sort([1,1,3,4,5], [1,2,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...
    >>> sort([1,1,3,4,5], [1,1,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    7.31617...
    >>> sort([1,2,3,4,5], [1,1,3,4,5], estimate="lowest")[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    6.73071...


    >>> sort([1,2,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> sort([2,1,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -2.32587...
    >>> sort([1,1,3,4,5], [1,2,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...
    >>> sort([1,1,3,4,5], [1,1,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> sort([1,2,3,4,5], [1,1,3,4,5], tau=False)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.58291...

    >>> sort([1,2,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> sort([2,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -4.0
    >>> sort([1,1,3,4,5], [1,2,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0
    >>> sort([1,1,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> sort([1,2,3,4,5], [1,1,3,4,5], tau=False, lambd=0.0001)[2]  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
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
    x0, y0, t0, Sx0, Sy0 = sort(x[:mid], y[:mid], estimate, tau, lambd)
    x1, y1, t1, Sx1, Sy1 = sort(x[mid:], y[mid:], estimate, tau, lambd)
    rx, ry, t = softmerge(x0, x1, y0, y1, Sx0, Sx1, Sy0, Sy1, estimate, tau, lambd)
    return rx, ry, t + t0 + t1, Sx0 + Sx1, Sy0 + Sy1
