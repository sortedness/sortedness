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
from torch import tensor

from sortedness.new.stats import findsigma_normal, findweight_normal, findbeta_cauchy, findweight_cauchy


def gaussian(n, kappa=5, pct=95):
    """
    >>> sum(gaussian(100))
    tensor(1., dtype=torch.float64)
    >>> round(float(sum(gaussian(5))), 2)
    0.95
    """
    sigma = findsigma_normal(pct, kappa)
    return tensor([findweight_normal(i, sigma) for i in range(n)])


def cauchy(n, kappa=5, pct=95):
    """
    >>> sum(cauchy(100))
    tensor(0.9975, dtype=torch.float64)
    >>> round(float(sum(cauchy(5))), 2)
    0.95
    """
    beta = findbeta_cauchy(pct, kappa)
    return tensor([findweight_cauchy(i, beta) for i in range(n)])
