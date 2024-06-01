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
from lange import gp, ap
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, weightedtau

from sortedness.new.weighting import cauchy

n = 30
kappas = n
step = 2
s = 10000
rng = np.random.default_rng(seed=0)
x = rng.uniform(0, 1, n)
for kappa in gp[5, 10, ..., kappas]:
    print(kappa / kappas)
    cau = list(cauchy(n, kappa=kappa))
    l = []
    for i in range(s):
        y = rng.uniform(0, 1, n)
        # tau = kendalltau(x, y)[0]
        wtau = weightedtau(x, y, weigher=lambda r: cau[r])[0]
        l.append(wtau)
    a = np.array(l)
    plt.hist(a, bins=ap[-1, -0.95, ..., 1], alpha=0.2, label=f"wtau 95% do peso nas primeiras {kappa} de {n} posições", edgecolor="k")

n = 30
s = 10000
rng = np.random.default_rng(seed=0)
l = []
x = rng.uniform(0, 1, n)
for i in range(s):
    y = rng.uniform(0, 1, n)
    tau = kendalltau(x, y)[0]
    l.append(tau)
a = np.array(l)
plt.hist(a, bins=ap[-1, -0.95, ..., 1], alpha=0.2, label=f"tau para {n} posições", edgecolor="k")

plt.legend()
plt.show()
