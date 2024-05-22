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
import numpy as np
from scipy.stats import kendalltau, spearmanr

rng = np.random.default_rng(seed=0)
l = []
for i in range(50000):
    c = 15 #int(rng.uniform(6, 50, 1)[0])
    x = rng.uniform(0, 1, c)
    y = rng.uniform(0, 1, c)
    tau = kendalltau(x, y)[0]
    rho = spearmanr(x, y)[0]
    # print(c, tau, rho)
    if tau > 0 or rho > 0:
        l.append((tau, rho))

a = np.array(l)
print(a)
for i in range(1, 40):
    st = (i - 1) / 40
    en = i / 40
    b = a[((abs(a[:, 0]) >= st) | (abs(a[:, 1]) >= st)) & ((abs(a[:, 0]) <= en) | (abs(a[:, 1]) <= en))][:30]
    if len(b) < 6:
        break
    print(len(b), st, en, kendalltau(b[:, 0], b[:, 1])[0])
