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


def argmergesort(r, idxs=None):
    """
    >>> argmergesort([1,2,3])
    [0, 1, 2]
    >>> argmergesort([4,3,2,1,0])
    [4, 3, 2, 1, 0]
    >>> argmergesort([2,4,3,1,0,9])
    [4, 3, 0, 2, 1, 5]

    :param idxs:
    :param r:
    :return:
    """
    if idxs is None:
        idxs = list(range(len(r)))
    n = len(idxs)
    if n == 1:
        return idxs[:1]
    mid = n // 2
    left = argmergesort(r, idxs[:mid])
    right = argmergesort(r, idxs[mid:])
    i = j = k = 0
    while i < len(left) and j < len(right):
        if r[left[i]] <= r[right[j]]:
            idxs[k] = left[i]
            i += 1
        else:
            idxs[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        idxs[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        idxs[k] = right[j]
        j += 1
        k += 1
    return idxs


def tau(r, idxs=None):
    """
    >>> argmergesort([1,2,3])
    [0, 1, 2]
    >>> argmergesort([4,3,2,1,0])
    [4, 3, 2, 1, 0]
    >>> argmergesort([2,4,3,1,0,9])
    [4, 3, 0, 2, 1, 5]

    :param idxs:
    :param r:
    :return:
    """
    if idxs is None:
        idxs = list(range(len(r)))
    n = len(idxs)
    if n == 1:
        return idxs[:1]
    mid = n // 2
    left = argmergesort(r, idxs[:mid])
    right = argmergesort(r, idxs[mid:])
    i = j = k = 0
    while i < len(left) and j < len(right):
        if r[left[i]] <= r[right[j]]:
            idxs[k] = left[i]
            i += 1
        else:
            idxs[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        idxs[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        idxs[k] = right[j]
        j += 1
        k += 1
    return idxs
