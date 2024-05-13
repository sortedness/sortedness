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


def weigh_ties(a, n, idxs, weigher, b=None):
    first = Uw = 0
    s = weigher(idxs[first])
    for i in range(1, n):
        if a[idxs[first]] != a[idxs[i]] or (b is not None and b[idxs[first]] != b[idxs[i]]):
            Uw += s * (i - first - 1)
            first = i
            s = 0
        w = weigher(idxs[i])
        s += w
    if first == 0:
        raise Exception(f"Constant vector provided: {a}")
    Uw += s * (n - first - 1)
    return Uw


def wtau(x, y, weigher=lambda r: 1 / (1 + r)):
    """

    >>> from scipy.stats import kendalltau, weightedtau
    >>> wtau([1,2,3,4,5], [1,2,3,4,5])
    1.0
    >>> wtau([1,2,3,4,5], [5,4,3,2,1])
    -1.0
    >>> round(wtau([1,2,3,4,5], [3,2,1,5,4]), 8), round(weightedtau([1,2,3,4,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.09854015, 0.09854015)
    >>> round(wtau([1,2,3,3,5], [3,2,1,5,4]), 8), round(weightedtau([1,2,3,3,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.03583477, 0.03583477)

    :param x:
    :param y:
    :param weigher: 
    :param idxs:
    :return:
    """

    """
    This function is based on code of Sebastian Vigna licensed as follows:    
    Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    
    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    idxs = list(range(len(x)))
    n = len(idxs)
    JtiesW = weigh_ties(x, n, idxs, weigher, y)  # Joint
    UtiesW = weigh_ties(x, n, idxs, weigher)  # x

    # Dw: Weight of disagreements
    temp = np.empty(n, dtype=np.intp)
    DW = np.zeros(1, dtype=np.float64)

    def weigh(offset, length):
        if length == 1:
            return weigher(idxs[offset])
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        residual = weigh(offset, length0)
        weight = weigh(middle, length1) + residual
        if y[idxs[middle - 1]] < y[idxs[middle]]:
            return weight

        # merging
        kk = ii = jj = 0
        while ii < length0 and jj < length1:
            if y[idxs[offset + ii]] <= y[idxs[middle + jj]]:
                temp[kk] = idxs[offset + ii]
                residual -= weigher(temp[kk])
                ii += 1
            else:
                temp[kk] = idxs[middle + jj]
                DW[0] += weigher(temp[kk]) * (length0 - ii) + residual
                jj += 1
            kk += 1

        idxs[offset + kk:offset + kk + length0 - ii] = idxs[offset + ii:offset + length0]
        idxs[offset:offset + kk] = temp[0:kk]
        return weight

    weigh(0, n)
    VtiesW = weigh_ties(y, n, idxs, weigher)  # y
    TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1)  # Total weight
    Aw = TW - UtiesW - VtiesW + JtiesW  # subtract single ties from total, and put joint ties back
    num = Aw - 2. * DW[0]
    den = np.sqrt(TW - UtiesW) * np.sqrt(TW - VtiesW)
    return min(1., max(-1., num / den))


def wsofttau(x, y, weigher=lambda r: 1 / (1 + r), lambd=1.0):
    """
    x should be sorted

    >>> from scipy.stats import kendalltau, weightedtau
    >>> wsofttau([1,2,3,4,5], [1,2,3,4,5], lambd=0.01)
    1.0
    >>> wsofttau([1,2,3,4,5], [5,4,3,2,1], lambd=0.01)
    -1.0
    >>> round(wsofttau([1,2,3,4,5], [3,2,1,5,4], lambd=0.01), 8), round(weightedtau([1,2,3,4,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.09854015, 0.09854015)
    >>> round(wsofttau([1,2,3,3,5], [3,2,1,5,4], lambd=0.01), 8), round(weightedtau([1,2,3,3,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.03583477, 0.03583477)

    :param x:
    :param y:
    :param weigher:
    :param idxs:
    :return:
    """

    """
    This function is based on code of Sebastian Vigna licensed as follows:    
    Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    
    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    
    2. Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.
    
    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    idxs = list(range(len(x)))
    n = len(idxs)
    JtiesW = weigh_ties(x, n, idxs, weigher, y)  # Joint
    UtiesW = weigh_ties(x, n, idxs, weigher)  # x

    # Dw: Weight of disagreements
    temp = np.empty(n, dtype=np.intp)
    DW = np.zeros(1, dtype=np.float64)

    def weigh(offset, length):
        if length == 1:
            return weigher(idxs[offset]), 0
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        leftw, a0 = weigh(offset, length0)
        rightw, a1 = weigh(middle, length1)
        weight, agreements = leftw + rightw, a0 + a1
        d = y[idxs[middle]] - y[idxs[middle - 1]]
        if d > 0:
            squashed = tanh(d / lambd) * tanh((y[idxs[middle]] - y[idxs[middle - 1]]) / lambd)
            return weight, agreements + length0 * length1 * squashed

        # merging
        kk = ii = jj = 0
        while ii < length0 and jj < length1:
            d = y[idxs[middle + jj]] - y[idxs[offset + ii]]
            squashed = tanh(d / lambd) * tanh((x[idxs[middle + jj]] - x[idxs[offset + ii]]) / lambd)
            if d >= 0:
                agreements += (length1 - jj) * squashed
                temp[kk] = idxs[offset + ii]
                w = weigher(temp[kk])
                leftw -= w
                ii += 1
            else:
                agreements += (length0 - ii) * squashed
                temp[kk] = idxs[middle + jj]
                w = weigher(temp[kk])
                rightw -= w
                jj += 1
                DW[0] += w * (length0 - ii) + leftw
            kk += 1

        idxs[offset + kk:offset + kk + length0 - ii] = idxs[offset + ii:offset + length0]
        idxs[offset:offset + kk] = temp[0:kk]
        return weight, agreements

    weight, agreements = weigh(0, n)
    VtiesW = weigh_ties(y, n, idxs, weigher)  # y
    TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1)  # Total weight
    Aw = TW - UtiesW - VtiesW + JtiesW  # subtract single ties from total, and put joint ties back
    num = Aw - 2. * DW[0]
    den = np.sqrt(TW - UtiesW) * np.sqrt(TW - VtiesW)
    return min(1., max(-1., num / den))
