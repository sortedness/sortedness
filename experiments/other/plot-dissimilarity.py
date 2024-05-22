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
from matplotlib import pyplot as plt
from numpy import argsort
from numpy.random import default_rng

from sortedness.misc.dissimilarity import matrix

rnd = default_rng(0)
X = rnd.random(size=(300, 2))
# print(X)
r = matrix(X, w="gaussian")[0]
# r = matrix(X)[0]
# print(list(sorted(r)))
s = 500 * (.01 + r) / max(r + .01)
# print(s)
# plt.scatter(X[:, 0], X[:, 1], s=20000, marker="+")
plt.scatter(X[:, 0], X[:, 1], s=s)
for i, l in enumerate(argsort(r)[:30]):
    plt.text(X[l, 0], X[l, 1], s=i, c="red", size=13)
plt.show()
