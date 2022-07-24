#  Copyright (c) 2022. Davi Pereira dos Santos
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

from random import shuffle

from sortedness.rank import rdist_by_index_lw

max = -1
for l in range(1000):
    print(l, f"last max:{max}")
    max = 0
    for _ in range(1000):
        lst = list(range(l))
        shuffle(lst)
        d = rdist_by_index_lw(lst, normalized=False)
        dnorm = rdist_by_index_lw(lst)
        if dnorm > max:
            max = dnorm

        if dnorm > 1:
            print(dnorm)
            raise Exception("bug")
