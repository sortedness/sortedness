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


def weigh_ties(a, b, n, idxs, weigher, joint=False):
    first = W = 0
    s = weigher(idxs[first])
    for i in range(1, n):
        beq = b[idxs[first]] != b[idxs[i]]
        w = weigher(idxs[i])
        if a[idxs[first]] != a[idxs[i]] or joint and beq:
            W += s * (i - first - 1)
            first = i
            s = w
        if not beq or joint:
            s += w
    if first == 0:
        raise Exception(f"Constant vector provided: {a}")
    W += s * (n - first - 1)
    return W


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
    >>> round(wtau([1,1,3,4,5], [5,5,3,2,1]), 8), round(weightedtau([1,1,3,4,5], [5,5,3,2,1], rank=False)[0], 8)
    (-1.0, -1.0)
    >>> round(wtau([1,1,3,4,5], [5,4,3,2,1]), 8), round(weightedtau([1,1,3,4,5], [5,4,3,2,1], rank=False)[0], 8)
    (-0.91420262, -0.91420262)
    >>> round(wtau([1,2,3,4,5], [5,5,3,2,1]), 8), round(weightedtau([1,2,3,4,5], [5,5,3,2,1], rank=False)[0], 8)
    (-0.91420262, -0.91420262)

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

    # print("wtau ", Aw, UtiesW,VtiesW,JtiesW,TW)

    return min(1., max(-1., num / den))


def wsofttau(x, y, weigher=lambda r: 1 / (1 + r), lambd=1.0):
    """
    x should be sorted

    >>> from scipy.stats import kendalltau, weightedtau
    >>> from numpy import array
    >>> a, b = array([1,2,3,4,5]), array([1,2,3,4,5])
    >>> wtau = lambda x,y: weightedtau(x, y, rank=False)[0]
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,4,5]), array([1,2,3,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,3,5]), array([1,1,3,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([1,1,3,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,4,5]), array([1,2,3,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,4,5]), array([1,2,2,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([1,2,2,4,5])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,4,5]), array([5,5,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([5,5,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,4,5]), array([5,4,4,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([5,4,4,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,1,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,2,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,5,5]), array([5,4,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,3,3,3,5]), array([5,5,3,2,1])
    >>> round(wsofttau(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,3,5]), array([3,2,5,5,4])
    >>> round(wsofttau(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True


    # >>> wsofttau(array([1,2,3,4,5]), array([1,2,3,4,5]), lambd=0.01)
    # 1.0
    # >>> wsofttau(array([1,2,3,4,5]), array([5,4,3,2,1]), lambd=0.01)
    # -1.0
    # >>> round(wsofttau(array([1,2,3,4,5]), array([3,2,1,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,4,5], [3,2,1,5,4], )[0], 8)
    # (0.09854015, 0.09854015)
    # >>> round(wsofttau(array([1,2,3,3,5]), array([3,2,1,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,1,5,4], )[0], 8)
    # (0.03583477, 0.03583477)
    # >>> round(wsofttau(array([1,2,3,3,5]), array([3,2,5,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,5,5,4], )[0], 8)
    # (0.41910331, 0.41910331)
    # >>> round(wsofttau(array([1,2,3,3,5]), array([3,3,5,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,5,5,4], )[0], 8)
    # (0.41910331, 0.41910331)

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
    AW = np.zeros(1, dtype=np.float64)
    DW = np.zeros(1, dtype=np.float64)

    def weigh(offset, length):
        # print(f"{x[idxs][offset:offset+length]}", f"{y[idxs][offset:offset+length]}")
        if length == 1:
            return weigher(idxs[offset]), 0
        leftlen = length // 2
        rightlen = length - leftlen
        middle = offset + leftlen
        leftw, a0 = weigh(offset, leftlen)
        rightw, a1 = weigh(middle, rightlen)
        weight, agreements = leftw + rightw, a0 + a1
        d = y[idxs[middle]] - y[idxs[middle - 1]]
        # if d > 0:
        #     lenmul = leftlen * rightlen
        #     squashed = tanh(d / lambd) * tanh((y[idxs[middle]] - y[idxs[middle - 1]]) / lambd)
        #     AW[0] += rightlen * leftw + leftlen * rightw
        #     return weight, agreements + lenmul * squashed

        # merging
        kk = ii = jj = 0
        while ii < leftlen and jj < rightlen:
            c = x[idxs[middle + jj]] - x[idxs[offset + ii]]
            d = y[idxs[middle + jj]] - y[idxs[offset + ii]]
            squashed = tanh(d / lambd) * tanh((x[idxs[middle + jj]] - x[idxs[offset + ii]]) / lambd)
            # if c == d == 0:
            #     current_rightlen = rightlen - jj
            #     agreements += current_rightlen * squashed
            #     temp[kk] = idxs[offset + ii]
            #     w = weigher(temp[kk])
            #     leftw -= w
            #     AW[0] += w * current_rightlen + rightw
            #     ii += 1
            #
            # kk += 1
            #
            # current_leftlen = leftlen - ii
            # # agreements += current_leftlen * squashed
            # temp[kk] = idxs[middle + jj]
            # w = weigher(temp[kk])
            # rightw -= w
            # DW[0] += w * current_leftlen + leftw
            # jj += 1
            if d >= 0:
                current_rightlen = rightlen - jj
                agreements += current_rightlen * squashed
                temp[kk] = idxs[offset + ii]
                w = weigher(temp[kk])
                leftw -= w
                ii += 1
                # if d > 0:
                AW[0] += w * current_rightlen + rightw
            else:
                current_leftlen = leftlen - ii
                agreements += current_leftlen * squashed
                temp[kk] = idxs[middle + jj]
                w = weigher(temp[kk])
                rightw -= w
                jj += 1
                DW[0] += w * current_leftlen + leftw
            kk += 1

        idxs[offset + kk:offset + kk + leftlen - ii] = idxs[offset + ii:offset + leftlen]
        idxs[offset:offset + kk] = temp[0:kk]
        # print(f"{x[idxs]=}", f"{y[idxs]=}")
        return weight, agreements

    weight, agreements = weigh(0, n)

    VtiesW = weigh_ties(y, n, idxs, weigher)  # y
    TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1)  # Total weight
    Aw = TW - UtiesW - VtiesW + JtiesW  # subtract single ties from total, and put joint ties back

    TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1)

    # num = (AW[0] - JtiesW) - DW[0]
    num = Aw - 2. * DW[0]

    print(x, y)
    print("wtau2", Aw, UtiesW, VtiesW, JtiesW, TW)

    den = np.sqrt((TW - UtiesW) * (TW - VtiesW))
    return min(1., max(-1., num / den))


"""
    [1 1 3 4 5] [5 4 3 2 1]
    wtau2 7.633333333333333 1.5 0.0 0.0 9.133333333333333
    (-1.0, -0.91420262)
"""


# def wsofttau2(x, y, weigher=lambda r: 1 / (1 + r), lambd=1.0):
def wsofttau2(x, y, weigher=lambda r: 1, lambd=1.0):
    """
    x should be sorted

    >>> from scipy.stats import kendalltau, weightedtau
    >>> from numpy import array    # >>> wtau = lambda x,y: weightedtau(x, y, rank=False)[0]
    >>> wtau = lambda x,y: weightedtau(x, y, rank=False, weigher=lambda r:1)[0]
    >>> a, b = array([1,2,2,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.8, -0.9486833)
    >>> a, b = array([1,1,2,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.9, -0.9486833)
    >>> a, b = array([1,2,2,4,5]), array([5,4,4,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.8, -1.0)
    >>> a, b = array([1,2,3,4,5]), array([5,4,4,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,1,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,2,2,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,1,3,5,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,3,3,3,5]), array([5,5,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,3,5]), array([3,2,5,5,4])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([1,2,2,4,5])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    True
    >>> a, b = array([1,2,3,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-1.0, -1.0)
    >>> a, b = array([1,1,3,4,5]), array([5,4,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.9, -0.9486833)
    >>> a, b = array([1,1,3,4,5]), array([5,5,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.8, -1.0)
    >>> a, b = array([1,2,3,4,5]), array([5,5,3,2,1])
    >>> round(wsofttau2(a, b, lambd=0.01), 8) , round(wtau(a, b), 8)
    (-0.9, -0.9486833)
    # >>> a, b = array([1,2,3,4,5]), array([1,2,3,4,5])
    # >>> round(wsofttau2(a, b, lambd=0.01), 8) == round(wtau(a, b), 8)
    # True
    # >>> a, b = array([1,1,3,4,5]), array([1,2,3,4,5])
    # >>> round(wsofttau2(a, b, lambd=0.01), 8), round(wtau(a, b), 8)
    # (0.9, 0.91420262)
    # >>> a, b = array([1,1,3,3,5]), array([1,1,3,4,5])
    # >>> round(wsofttau2(a, b, lambd=0.01), 8), round(wtau(a, b), 8)
    # True
    # >>> a, b = array([1,2,3,4,5]), array([1,1,3,4,5])
    # >>> round(wsofttau2(a, b, lambd=0.01), 8), round(wtau(a, b), 8)
    # True
    # >>> a, b = array([1,2,2,4,5]), array([1,2,3,4,5])
    # >>> round(wsofttau2(a, b, lambd=0.01), 8), round(wtau(a, b), 8)
    # True
    >>> a, b = array([1,2,2,4,5]), array([1,2,2,4,5])
    >>> round(wsofttau2(a, b, lambd=0.01), 8), round(wtau(a, b), 8)
    True


    # >>> wsofttau2(array([1,2,3,4,5]), array([1,2,3,4,5]), lambd=0.01)
    # 1.0
    # >>> wsofttau2(array([1,2,3,4,5]), array([5,4,3,2,1]), lambd=0.01)
    # -1.0
    # >>> round(wsofttau2(array([1,2,3,4,5]), array([3,2,1,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,4,5], [3,2,1,5,4], )[0], 8)
    # (0.09854015, 0.09854015)
    # >>> round(wsofttau2(array([1,2,3,3,5]), array([3,2,1,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,1,5,4], )[0], 8)
    # (0.03583477, 0.03583477)
    # >>> round(wsofttau2(array([1,2,3,3,5]), array([3,2,5,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,5,5,4], )[0], 8)
    # (0.41910331, 0.41910331)
    # >>> round(wsofttau2(array([1,2,3,3,5]), array([3,3,5,5,4]), lambd=0.01), 8), round(kendalltau([1,2,3,3,5], [3,2,5,5,4], )[0], 8)
    # (0.41910331, 0.41910331)

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
    JtiesW = weigh_ties(x, y, n, idxs, weigher, True)  # Joint
    UtiesW = weigh_ties(x, y, n, idxs, weigher)  # x

    # Dw: Weight of disagreements
    temp = np.empty(n, dtype=np.intp)
    AW = np.zeros(1, dtype=np.float64)
    DW = np.zeros(1, dtype=np.float64)

    def weigh(offset, length):
        # print(f"{x[idxs][offset:offset+length]}", f"{y[idxs][offset:offset+length]}")
        if length == 1:
            return weigher(idxs[offset]), 0
        leftlen = length // 2
        rightlen = length - leftlen
        middle = offset + leftlen
        leftw, a0 = weigh(offset, leftlen)
        rightw, a1 = weigh(middle, rightlen)
        weight, agreements = leftw + rightw, a0 + a1
        d = y[idxs[middle]] - y[idxs[middle - 1]]
        # if d >= 0:
        #     lenmul = leftlen * rightlen
        #     squashed = tanh(d / lambd) * tanh((y[idxs[middle]] - y[idxs[middle - 1]]) / lambd)
        #     AW[0] += rightlen * leftw + leftlen * rightw
        #     return weight, agreements + lenmul * squashed

        # merging
        kk = ii = jj = 0
        while ii < leftlen and jj < rightlen:
            c = x[idxs[middle + jj]] - x[idxs[offset + ii]]
            d = y[idxs[middle + jj]] - y[idxs[offset + ii]]
            # squashed = tanh(c / lambd) * tanh(d / lambd)
            squashed = 1 - abs(tanh(c / lambd) - tanh(d / lambd))  # .sqrt()
            if d >= 0:
                current_rightlen = rightlen - jj
                agreements += current_rightlen * squashed
                # print("a")
                temp[kk] = idxs[offset + ii]
                w = weigher(temp[kk])
                leftw -= w
                ii += 1
                # if c > 0:
                AW[0] += w * current_rightlen + rightw
            else:
                current_leftlen = leftlen - ii
                # print("d", agreements, squashed, current_leftlen)
                agreements += current_leftlen * squashed
                temp[kk] = idxs[middle + jj]
                w = weigher(temp[kk])
                rightw -= w
                jj += 1
                DW[0] += w * current_leftlen + leftw
            kk += 1

        idxs[offset + kk:offset + kk + leftlen - ii] = idxs[offset + ii:offset + leftlen]
        idxs[offset:offset + kk] = temp[0:kk]
        # print(f"{x[idxs]=}", f"{y[idxs]=}")
        return weight, agreements

    weight, agreements = weigh(0, n)

    # VtiesW = weigh_ties(y, x, n, idxs, weigher)  # y
    # TW = sum(weigher(idxs[i]) for i in range(n)) * (n - 1) / 2
    # num = AW[0] + JtiesW - DW[0] + UtiesW - VtiesW
    # den = TW
    # print(x, y, f"→→ {AW=}  {JtiesW=}  {UtiesW=}   {VtiesW=}  →→  {DW=}  {TW=}")
    # print(x, y, f"→→ {agreements=}")
    # # return min(1., max(-1., num / den))
    return agreements / (n * (n - 1) / 2)

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    # con = ntie - dis - xtie - ytie
    # con_minus_dis = tot - xtie - ytie + ntie - 2 * dis

    # tot = s * (n - 1)
    # tau = ((tot - (v + u - t)) - 2. * exchanges_weight[0]) / sqrt(tot - u) / sqrt(tot - v)
