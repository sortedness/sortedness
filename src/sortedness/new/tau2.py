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
        i = j = k = 0
        while j < length0 and k < length1:
            if y[idxs[offset + j]] <= y[idxs[middle + k]]:
                temp[i] = idxs[offset + j]
                residual -= weigher(temp[i])
                j += 1
            else:
                temp[i] = idxs[middle + k]
                DW[0] += weigher(temp[i]) * (length0 - j) + residual
                k += 1
            i += 1

        idxs[offset + i:offset + i + length0 - j] = idxs[offset + j:offset + length0]
        idxs[offset:offset + i] = temp[0:i]
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

    >>> from scipy.stats import kendalltau, weightedtau
    >>> wsofttau([1,2,3,4,5], [1,2,3,4,5])
    1.0
    >>> wsofttau([1,2,3,4,5], [5,4,3,2,1], lambd=0.01)
    -1.0
    >>> round(wsofttau([1,2,3,4,5], [3,2,1,5,4], lambd=0.01), 8), round(weightedtau([1,2,3,4,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.09854015, 0.09854015)
    >>> round(wsofttau([1,2,3,3,5], [3,2,1,5,4], lambd=0.01), 8), round(weightedtau([1,2,3,3,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.03583477, 0.03583477)
    >>> round(wsofttau([1,2,3,3,5], [3,2,5,1,4], lambd=0.01), 8), round(weightedtau([1,2,3,3,5], [3,2,5,1,4], rank=False)[0], 8)
    (0.05469518, 0.05469518)
    >>> round(wsofttau([1,2,3,4,5], [3,2,1,1,4], lambd=0.01), 8), round(weightedtau([1,2,3,4,5], [3,2,1,1,4], rank=False)[0], 8)
    (-0.31496878, -0.31496878)
    >>> round(wsofttau([1,2,3,3,5], [3,2,1,1,4], lambd=0.01), 8), round(weightedtau([1,2,3,3,5], [3,2,1,1,4], rank=False)[0], 8)
    (-0.32553606, -0.32553606)
    >>> wsofttau([1,2,3,4,5], [1,2,3,4,5])
    1.0
    >>> round(wsofttau([1,2,3,4,5], [5,4,3,2,1]), 8)
    -0.72037107
    >>> round(wsofttau([1,2,3,4,5], [3,2,1,5,4]), 8), round(weightedtau([1,2,3,4,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.31139272, 0.09854015)
    >>> round(wsofttau([1,2,3,3,5], [3,2,1,5,4]), 8), round(weightedtau([1,2,3,3,5], [3,2,1,5,4], rank=False)[0], 8)
    (0.24012676, 0.03583477)

    :param lambd:
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
    temp = np.empty(n, dtype=np.intp)
    AW = [0]  # np.zeros(1, dtype=np.float64)
    DW = [0]  # np.zeros(1, dtype=np.float64)

    def weigh(offset, length):
        if length == 1:
            return weigher(idxs[offset])
        length0 = length // 2
        length1 = length - length0
        middle = offset + length0
        residual0 = weigh(offset, length0)
        residual1 = weigh(offset, length1)
        weight = residual0 + residual1
        if y[idxs[middle - 1]] < y[idxs[middle]]:
            upper_deltax = x[idxs[middle]] - x[idxs[middle - 1]]
            upper_deltay = y[idxs[middle]] - y[idxs[middle - 1]]
            soft = abs(np.tanh(upper_deltax / lambd) * np.tanh(upper_deltay / lambd))
            AW[0] += weight * soft
            return weight

        # merging
        i = j = k = 0
        while j < length0 and k < length1:
            upper_deltax = x[idxs[middle + k]] - x[idxs[offset + j]]
            deltay = y[idxs[middle + k]] - y[idxs[offset + j]]
            soft = abs(np.tanh(upper_deltax / lambd) * np.tanh(deltay / lambd))
            if deltay >= 0:
                temp[i] = idxs[offset + j]
                w = weigher(temp[i])
                residual0 -= w
                AW[0] += (w * (length1 - k) + residual1) * soft
                j += 1
            else:
                # lower_deltax = x[idxs[middle + k]] - x[idxs[offset + j]]
                upper_deltax = x[idxs[middle + k]] - x[idxs[offset + length0 - 1]]
                upper_deltay = y[idxs[middle + k]] - y[idxs[offset + length0 - 1]]
                soft = abs(np.tanh(upper_deltax / lambd) * np.tanh(upper_deltay / lambd))
                temp[i] = idxs[middle + k]
                w = weigher(temp[i])
                residual1 -= w
                DW[0] += (w * (length0 - j) + residual0) * soft
                k += 1
            i += 1

        idxs[offset + i:offset + i + length0 - j] = idxs[offset + j:offset + length0]
        idxs[offset:offset + i] = temp[0:i]
        return weight

    weigh(0, n)
    VtiesW = weigh_ties(y, n, idxs, weigher)  # y
    TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1)  # Total weight
    Aw = TW - DW[0] - UtiesW - VtiesW + JtiesW  # to know agreements: subtract disagreements, single ties from total, and put joint ties back
    num = Aw - DW[0]  # single tie counts as zero (not agreement nor disagreement)
    den = np.sqrt(TW - UtiesW) * np.sqrt(TW - VtiesW)
    return min(1., max(-1., (AW[0] - DW[0]) / (AW[0] + DW[0])))
    # return min(1., max(-1., num / den))
