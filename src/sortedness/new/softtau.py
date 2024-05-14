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


def softmerge(x0, x1, y0, y1, mx0, mx1, my0, my1, tau=True, lambd=1.0):
    """
    sorts on x

    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 15, 6, 15, tau=True, lambd=0.0000001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    >>> softmerge([1,2,3], [4,5,6], [1,2,3], [4,5,6], 6, 15, 6, 15, tau=True, lambd=0.0000001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    >>> softmerge([1,2,3], [1,3,5], [5,6,7], [8,9,10], 6, 15, 6, 15, lambd=0.0000001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    >>> softmerge([1,2,3], [4,5,6], [1,2,3], [4,5,6], 6, 15, 6, 15, lambd=0.0000001)  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    ([1, 1, 3, 5, 20, 300], [5, 8, 9, 10, 6, 7], 1.3829..., 5, 0, 13, 0)

    :param x0:
    :param x1:
    :param y0:
    :param y1:
    :param lambd:
    :return:
    """
    if not x0:
        return x1, y1, 0, mx0, mx1, my0, my1
    if not x1:
        return x0, y0, 0, mx0, mx1, my0, my1
    if x0[0] <= x1[0]:
        mx0_, my0_ = mx0 - x0[0], my0 - y0[0]
        rx, ry, t, mx0, mx1, my0, my1 = softmerge(x0[1:], x1, y0[1:], y1, mx0_, mx1, my0_, my1, tau, lambd)
        l = len(x1)
        tanx = tanh((mx1 / l - x0[0]) / lambd)
        tany = tanh((my1 / l - y0[0]) / lambd)
        print(tany, f"{my1}/{l} - {y0[0]}")
        dt = (tanx * tany) if tau else ((tanx - tany) ** 2)
        return x0[:1] + rx, y0[:1] + ry, t + l * dt, mx0, mx1, my0, my1
    else:
        mx1_, my1_ = mx1 - x1[0], my1 - y1[0]
        rx, ry, t, mx0, mx1, my0, my1 = softmerge(x0, x1[1:], y0, y1[1:], mx0, mx1, my0, my1_, tau, lambd)
        l = len(x0)
        tanx = tanh((x1[0] - mx0 / l) / lambd)
        tany = tanh((y1[0] - my0 / l) / lambd)
        dt = (tanx * tany) if tau else ((tanx - tany) ** 2)
        return x1[:1] + rx, y1[:1] + ry, t + l * dt, mx0, mx1, my0, my1


def sort(x, y, tau=True, lambd=0.1):
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

    """
    xl, yl = len(x), len(y)
    if xl == 1:
        return x, y, 0, x[0], y[0]
    mid = xl // 2
    x0, y0, t0, mx0, my0 = sort(x[:mid], y[:mid])
    x1, y1, t1, mx1, my1 = sort(x[mid:], y[mid:])
    rx, ry, t, mx0, mx1, my0, my1 = softmerge(x0, x1, y0, y1, mx0, mx1, my0, my1, tau, lambd)
    return rx, ry, t + t0 + t1, mx0 + mx1, my0 + my1
