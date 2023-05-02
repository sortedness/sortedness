#cython: boundscheck=False, wraparound=False, nonecheck=False

"""
Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
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

import cython
from cpython cimport bool
from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy.math cimport INFINITY
from numpy.math cimport NAN
from numpy cimport ndarray, int64_t, float64_t, intp_t
import warnings
import numpy as np
cimport scipy.special.cython_special as cs
from cython.parallel import prange
from libc.math cimport sqrt
from libc.stdio cimport printf

np.import_array()

ctypedef fused ordered:
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

cpdef float64_t weigh(  intp_t offset,
                        intp_t length,
                        intp_t[::1] rank,
                        intp_t[::1] perm,
                        intp_t[::1] temp,
                        float64_t[::1] exchanges_weight,
                        ordered[::1] x,
                        ordered[::1] y
                    ) nogil:
    cdef intp_t length0, length1, middle, i, j, k
    cdef float64_t weight, residual

    if length == 1:
        return 1./(1 + rank[perm[offset]])
    length0 = length // 2
    length1 = length - length0
    middle = offset + length0
    residual = weigh(offset, length0, rank, perm, temp, exchanges_weight, x, y)
    weight = weigh(middle, length1, rank, perm, temp, exchanges_weight, x, y) + residual
    if y[perm[middle - 1]] < y[perm[middle]]:
        return weight

    # merging
    i = j = k = 0

    while j < length0 and k < length1:
        if y[perm[offset + j]] <= y[perm[middle + k]]:
            temp[i] = perm[offset + j]
            residual -= 1./(1 + rank[temp[i]])
            j += 1
        else:
            temp[i] = perm[middle + k]
            exchanges_weight[0] += 1./(1 + rank[temp[i]]) * (length0 - j) + residual
            k += 1
        i += 1

    perm[offset+i:offset+i+length0-j] = perm[offset+j:offset+length0]
    perm[offset:offset+i] = temp[0:i]
    return weight


cpdef float64_t _weightedrankedtau( ordered[::1] x, ordered[::1] y, intp_t[::1] rank,
                                    int64_t n,
                                    intp_t[::1] perm,
                                    float64_t[::1] exchanges_weight,
                                    intp_t[::1] temp
                                ) nogil:
    cdef ordered[::1] y_local = y
    cdef intp_t i, first
    cdef float64_t t, u, v, w, s, sq, tot, tau
    exchanges_weight[0] = 0

    # if rank is None:
    #     # To generate a rank array, we must first reverse the permutation
    #     # (to get higher ranks first) and then invert it.
    #     rank = np.empty(n, dtype=np.intp)
    #     rank[...] = perm[::-1]
    #     _invert_in_place(rank)

    cdef intp_t[::1] rank_local = rank

    # weigh joint ties
    first = 0
    t = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]] or y[perm[first]] != y[perm[i]]:
            t += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    t += s * (n - first - 1)

    # weigh ties in x
    first = 0
    u = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if x[perm[first]] != x[perm[i]]:
            u += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    u += s * (n - first - 1)
    # if first == 0: # x is constant (all ties)
    #     return np.nan

    # weigh discordances
    weigh(0, n, rank, perm, temp, exchanges_weight, x, y)

    # weigh ties in y
    first = 0
    v = 0
    w = 1./(1 + rank[perm[first]])
    s = w
    sq = w * w

    for i in range(1, n):
        if y[perm[first]] != y[perm[i]]:
            v += s * (i - first - 1)
            first = i
            s = sq = 0

        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    v += s * (n - first - 1)
    # if first == 0: # y is constant (all ties)
    #     return np.nan

    # weigh all pairs
    s = sq = 0
    for i in range(n):
        w = 1./(1 + rank[perm[i]])
        s += w
        sq += w * w

    tot = s * (n - 1)

    tau = ((tot - (v + u - t)) - 2. * exchanges_weight[0]) / sqrt(tot - u) / sqrt(tot - v)
    return min(1., max(-1., tau))


# Inverts a permutation in place [B. H. Boonstra, Comm. ACM 8(2):104, 1965].
cpdef _invert_in_place(intp_t[:] perm):
    cdef intp_t n, i, j, k
    for n in range(len(perm)-1, -1, -1):
        i = perm[n]
        if i < 0:
            perm[n] = -i - 1
        else:
            if i != n:
                k = n
                while True:
                    j = perm[i]
                    perm[i] = -k - 1
                    if j == n:
                        perm[n] = i
                        break

                    k = i
                    i = j


# def parwtau(ordered[::1] x, ordered[::1] y, intp_t[:,:] ranks):
#     cdef int64_t n = np.int64(len(x))
#     cdef int64_t m = np.int64(len(ranks[0]))
#     cdef float64_t[::1] ret = np.empty(m, dtype=np.float64)
#     cdef intp_t[n] perm
#     # cdef float64_t[::1] exchanges_weight
#     cdef int[n] temp
#     cdef int[n] perm0 = np.lexsort((y, x))
#     cdef float64_t[::1] exchanges_weight = np.zeros(1, dtype=np.float64)
#     cdef int[n] temp0 = np.empty(n, dtype=np.intp)
#     cdef int64_t i, r
#     for i in prange(len(ranks), nogil=True):
#         for j in range(n):
#             perm[j] = perm0[j]
#             temp[j] = temp0[j]
#         exchanges_weight[0] = 0
#         ret[i] = _weightedrankedtau(x, y, ranks[:, i], n, perm, exchanges_weight, temp)
#     return ret


