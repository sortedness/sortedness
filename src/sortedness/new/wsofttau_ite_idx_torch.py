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
import torch
from torch import sum, tanh, tensor


# raise Exception(f"not working")
def merge(x, y, w, idx, sx, sy, sw, estimate, tau, lambd, tmp, start, mid, end):
    """
    sorts on x
    """
    # todo: criar opção de quasitau que lida com ties da forma que acho mais correta?
    # todo: add parameter to indicate the degree of approximation; from total heuristic (1) to no heuristic (0)
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
                tanx = tanh((x[idx[j:end]] - x[idx[i]]) / lambd)
                tany = tanh((y[idx[j:end]] - y[idx[i]]) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[i]] + w[idx[j:end]]) / 2
                t += sum(weight * dt)
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
                tanx = tanh((x[idx[j]] - x[idx[i:mid]]) / lambd)
                tany = tanh((y[idx[j]] - y[idx[i:mid]]) / lambd)
                dt = (tanx * tany) if tau else -((tanx - tany) ** 2)
                weight = (w[idx[j]] + w[idx[i:mid]]) / 2
                t += sum(weight * dt)
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
        tmp[k:end] = idx[i:mid]
        idx[k:end] = tmp[i:mid]
    idx[start:k] = tmp[start:k]
    return t


# todo: tau=True could use the correct denominator, provavlmente tem que criar outro acumulador
#   den = sum(tana ** 2 * sw) * sum(tanb ** 2 * sw)

def wsoft_sort(x, y, w, idx, estimate="average", tau=True, lambd=1.0):
    """y should be sorted

    >>> from torch import tensor
    >>> a = list(reversed(range(0, 5)))
    >>> w = tensor(a, dtype=torch.float32) / sum(tensor(a, dtype=torch.float32))
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest"))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.77947...
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest"))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.34044...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest"))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest"))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="highest"))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.57646...

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65...

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), tau=False))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), tau=False))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.4
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.0
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.35
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w*0+1, tensor([0,1,2,3,4]),  tau=False, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -1.0

    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    2.0
    >>> float(wsoft_sort(tensor([2.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.3
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> float(wsoft_sort(tensor([1.0,1,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.65
    >>> float(wsoft_sort(tensor([1.0,2,3,4,5]), tensor([1.0,1,3,4,5]), w*0+1, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
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
    n = len(idx)
    if n < 2:
        return 0
    tmp = torch.clone(idx)
    sx, sy, sw = x.clone(), y.clone(), w.clone()
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


def wsoft_tau(x, y, w=None, idx=None, estimate="average", tau=True, lambd=1.0, normalized=True):
    """
    >>> from numpy import array
    >>> from torch import tensor
    >>> a = list(reversed(range(0, 5)))
    >>> w = tensor(a, dtype=torch.float32) / sum(tensor(a, dtype=torch.float32))
    >>> float(wsoft_tau(tensor([1.0,2,3,4,5]), tensor([1.0,2,3,4,5]), w, tensor([0,1,2,3,4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    1.0
    >>> a = list(range(100))
    >>> b = list(reversed(a))
    >>> from random import shuffle
    >>> import random
    >>> random.seed(0)
    >>> w = tensor(b, dtype=torch.float32) / sum(tensor(b, dtype=torch.float32))
    >>> idx = a.copy()
    >>> float(wsoft_tau(tensor(a, dtype=torch.float32), tensor(a, dtype=torch.float32), w, tensor(idx.copy()), estimate="highest", lambd=0.0001))
    1.0
    >>> float(wsoft_tau(tensor(b[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate="highest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> float(wsoft_tau(tensor(b[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate="average", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> float(wsoft_tau(tensor(b[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate="lowest", lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> float(wsoft_tau(tensor(a[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.99999...
    >>> float(wsoft_tau(tensor(b[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate=None, lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.99999...
    >>> float(wsoft_tau(tensor(b[:10], dtype=torch.float32), tensor(a[:10], dtype=torch.float32), w[:10], tensor(idx.copy()[:10]), estimate="highest", lambd=0.0001, normalized=False))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.19090...
    >>> float(wsoft_tau(tensor(b, dtype=torch.float32), tensor(a, dtype=torch.float32), w, tensor(idx.copy()), estimate="highest", lambd=0.0001))
    -1.0
    >>> s = 0
    >>> for i in range(100):
    ...     shuffle(b)
    ...     s += wsoft_tau(tensor(b), tensor(a), w, tensor(idx.copy()), estimate="highest", lambd=0.0001)
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.00075...
    >>> random.seed(1000)
    >>> s = 0
    >>> c = 0
    >>> from sortedness.new.quality.measure.pairwise import softtau
    >>> from torch import from_numpy
    >>> for i in range(100):
    ...     shuffle(b)
    ...     r1 = wsoft_tau(tensor(b), tensor(a), w, tensor(idx.copy()), estimate=None)
    ...     r2 = softtau(from_numpy(array(b)), from_numpy(array(a)), w)
    ...     if abs(r1 - r2) < 0.01:
    ...         c += 1
    ...     s += r1
    >>> c / 100
    1.0
    >>> s / 100  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    -0.0005...
    >>> a = list(range(6))
    >>> b = list(reversed(a))
    >>> w = tensor(array(b)) / tensor(sum(b))
    >>> b = a.copy()
    >>> b[0], b[1] = a[1], a[0]
    >>> float(wsoft_tau(tensor(b[:4]), tensor(a[:4]), w[:4], tensor(idx.copy()[:4]), lambd=0.0001, normalized=False))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.53333...
    >>> float(wsoft_tau(tensor(b[:4]), tensor(a[:4]), w[:4], tensor(idx.copy()[:4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
    0.57142...
    >>> float(wsoft_tau(tensor(b[:4]), tensor(a[:4]), w[:4]*0+1, tensor(idx.copy()[:4]), lambd=0.0001))  # doctest:+ELLIPSIS +NORMALIZE_WHITESPACE
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
    if idx is None:
        idx = tensor(list(range(len(x))))
    if w is None:
        w = tensor([1.0 / len(x)] * len(x))
    total_weight = sum(w) if normalized else 1
    s = wsoft_sort(x, y, w, idx, estimate=estimate, tau=tau, lambd=lambd)
    n = len(x)
    # return min(1., max(-1., s / (n - 1) * 2 / total_weight))
    return s / (n - 1) * 2 / total_weight
