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
from itertools import chain


def sort_(l, first=True):
    print(f"------------------- {l}")
    lsts = []
    c = 0
    i = 0
    while i < len(l):
        print(f" {i=} poe [\t{l[i]} · ", end="\t")
        lsts.append([l[i]])
        j = i + 1
        if j < len(l):
            while l[j] > l[j - 1]:
                print(f"\t{l[j]}", end="\t")
                lsts[c].append(l[j])
                j += 1
                if j == len(l):
                    break
        print("]\n", lsts)
        c += 1
        i = j
    if len(lsts) == len(l):
        # match len(l):
        #     case 1:
        #         return l[0]
        #     case 2:
        #         return l if l[0] < l[1] else [l[1], l[0]]
        return lsts if first else list(chain(*lsts))

    return sort_(lsts, first=False)


def sort(l, first=True):
    if first:
        l2 = sort_(l, first=True)
        l4 = []
        for l3 in l2:
            print(l3)
            if isinstance(l3[0], list):
                l4.extend(l3)
            else:
                l4.append(l3)
    else:
        l4 = l
    if len(l4)==1:
        l4[0]
    rr = []
    for a, b in zip(*[iter(l4)] * 2):
        i = j = 0
        r = []
        while True:
            if a[i] < b[j]:
                r.append(a[i])
                i += 1
                if i == len(a):
                    r.extend(b[i + 1:])
                    break
            else:
                r.append(b[j])
                j += 1
                if j == len(b):
                    r.extend(a[i + 1:])
                    break
        rr.append(r)
    print(rr)
    return sort(rr, first=False)


#     return list(chain(*[[sort(a), sort(b)] if a < b else [sort(b), sort(a)] for a, b in zip(*[iter(l2)] * 2)]))


s = sort([5, 7, 6, 4, 8, 10, 3, 6, 9])
print(s)
# print("====================================")
# s = sort([1, 2, 3, 4, 5, 6, 7])
# print(s)
# print("====================================")
# s = sort([7, 6, 5, 4, 3, 2, 1])
# print(s)
