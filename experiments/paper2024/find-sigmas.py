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
from pprint import pprint

from lange import ap
from scipy.stats import halfnorm

# Just a workaround while we don't have time for implementing a better approach in determining sigma from kappa.

old = -1
precalculated_sigma = {}
for pct in ap[1, 2, 5, 10, 12.5, 15, 20, 25, 30, 33.33, 40, 50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
    precalculated_sigma[pct] = None
    p = pct / 100
    for sigma in ap[.1, .2, ..., 300]:
        kappa = int(halfnorm.ppf(p, 0, sigma))
        if kappa == old or kappa == 0:
            continue
        old = kappa

        # print(f"{pct=}\t{kappa=:0.4f}\t{sigma=:0.4f}\ts={halfnorm.cdf(kappa, scale=sigma):0.4f}")
        if precalculated_sigma[pct] is None:
            precalculated_sigma[pct] = {}
        precalculated_sigma[pct][kappa] = sigma
    if precalculated_sigma[pct] is None:
        del precalculated_sigma[pct]

pprint(precalculated_sigma)

