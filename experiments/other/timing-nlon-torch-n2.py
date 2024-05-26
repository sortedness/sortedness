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

from timeit import timeit

import torch
from numpy import array

from sortedness.new.quality.measure.pairwise import softtau
from sortedness.new.wsofttau_ite_idx import wsoft_tau
from sortedness.new.wsofttau_ite_idx_torch import wsoft_tau as wsoft_taut

lst = list(range(20000))
x = array(lst)
y = array(lst)
#y = array(list(reversed(lst)))
w = array(list(reversed(lst)))
idx = array(lst)
tx = torch.tensor(lst)
ty = torch.tensor(lst)
#ty = torch.tensor(list(reversed(lst)))
tw = torch.tensor(list(reversed(lst)))
tidx = torch.tensor(lst)

# warmup
t = timeit(lambda: wsoft_taut(tx, ty, tw, tidx, "average", True, 0.0001, True), number=1)
t = timeit(lambda: softtau(tx, ty, tw, 0.0001), number=1)
t = timeit(lambda: wsoft_tau(x, y, w, idx, "average", True, 0.0001, True), number=1)

print("nlogn torch")
t = timeit(lambda: wsoft_taut(tx, ty, tw, tidx, "average", True, 0.0001, True), number=1)
print(f"{t:5.3f} s")

print("")
print("n² torch")
t = timeit(lambda: softtau(tx, ty, tw, 0.0001), number=1)
print(f"{t:5.3f} s")

print("")
print("nlogn meio-python meio-numpy")
t = timeit(lambda: wsoft_tau(x, y, w, idx, "average", True, 0.0001, True), number=1)
print(f"{t:5.3f} s")
