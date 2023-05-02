#  Copyright (c) 2023. Davi Pereira dos Santos
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

"""
README
======

Building:
CYTHONIZE=1 ; poetry run cythonize -a -i ./src/sortedness/wtau/wtau.pyx
"""

from itertools import repeat, tee

import pathos.multiprocessing as mp
from numpy import lexsort, array, empty

from .wtau import _weightedrankedtau  # Source wtau.pyx becomes wtau.c and shared so


def parwtau(scoresX, scoresX_, npoints, R=True, parallel=True, **kwargs):
    """**kwargs is for ProcessingPool, e.g.: npcus=8"""
    npositions = len(scoresX)
    if len(scoresX.shape) != 2:
        lex = lexsort((scoresX_, scoresX))
        perms_source = (lex.copy() for i in range(npoints))
    # else:
    #     perms_source = (lexsort((scoresX_[i], scoresX[i])) for i in range(npoints))
    perms, perms_for_r = tee(perms_source)
    add = R is True

    def genR(perms_gen, scores):
        for i in range(npoints):
            # print("s", i)
            p = next(perms_gen)
            # if R is True or R is None:
            #     r = p.copy()
            #     # print(r)
            #     _invert_in_place(r)
            if R is not False:
                r = R[:, i]
            # else:
            #     r = arange(npositions, dtype=intp)
            yield r

    exchanges_weights = (array([0.0]) for i in range(npoints))
    temps = (empty(npositions, dtype=int) for i in range(npoints))

    pmap = mp.ProcessingPool(**kwargs).imap if parallel else map
    if not add:
        jobs = repeat(scoresX), repeat(scoresX_), genR(perms_for_r, scoresX), repeat(npositions), perms, exchanges_weights, temps
        return array(list(pmap(_weightedrankedtau, *jobs)))

    # def f(scoresX, scoresX_, rank, rank_, perm, perm_, exchanges_weight, temp):
    #     return (_weightedrankedtau(scoresX, scoresX_, rank, npositions, perm, exchanges_weight, temp) +
    #             _weightedrankedtau(scoresX_, scoresX, rank_, npositions, perm_, exchanges_weight, temp)) / 2
    #
    # scores = lambda M: (M[i] for i in range(npoints))
    # if len(scoresX_.shape) == 2:
    #     perms_source_ = (lexsort((scoresX[i], scoresX_[i])).copy() for i in range(npoints))
    # else:
    #     lex = lexsort((scoresX, scoresX_))
    #     perms_source_ = (lex.copy() for i in range(npoints))
    # perms_, perms_for_r_ = tee(perms_source_)
    # jobs = scores(scoresX), scores(scoresX_), genR(perms_for_r, scoresX), genR(perms_for_r_, scoresX_), perms, perms_, exchanges_weights, temps
    # return array(list(pmap(f, *jobs)))


parwtau.isparwtau = True
