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

import matplotlib.pyplot as plt
from lange import ap
from sympy.utilities.iterables import multiset_permutations

from sortedness.rank import rdist_by_index_lw

ds = {}
for l in ap[5, 6, ..., 8]:
    print(l)
    lst = list(ap[0, 1, ..., l])
    ds[l] = []
    for p in multiset_permutations(lst):
        ds[l].append(rdist_by_index_lw(p, normalized=False))
        # ds[l].append(rank_based_dist__by_index(p, normalized=True))
    ds[l].sort()
    # plt.plot(ds[l])

plt.hist(ds.values(), bins=25, log=True, histtype="barstacked")

# plt.scatter(arr[:, 0], arr[:, 1], s=pen_width, c=colors)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
