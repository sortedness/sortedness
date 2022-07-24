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

from numpy import eye
from numpy.linalg import norm
from numpy.random import randint, shuffle

from sortedness.rank import rdist_by_index_lw, stress

old = 0
for l in range(3000000000000000):
    d = rdist_by_index_lw(list(range(l - 1, -1, -1)), normalized=False)
    # dnorm = rank_based_dist__by_index(list(range(l - 1, -1, -1)))
    print(l, "\t", d, "\t", d - old)
    old = d

#
# import numpy as np
# projected = np.array([[1,2], [-1,3], [0,1]])
#
# print(norm(projected-projected[0], axis=1, keepdims=True))
#
# import numpy as np
#
# mean = (1, 2)
# cov = eye(2)
# rng = np.random.default_rng(seed=6)
# original = rng.multivariate_normal(mean, cov, size=5)
# print(stress(original, original))
#
# r = np.array(list(range(10)))
# np.random.seed(0)
# shuffle(r)
# print(r)
# print(r[np.argsort(r)])
