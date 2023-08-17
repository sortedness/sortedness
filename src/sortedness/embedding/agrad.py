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
import autograd.numpy as np
from autograd import elementwise_grad as egrad


def sigm(x):
    return 0.5 * (np.tanh(x / 2.) + 1)


def pd(x):
    n = x.shape[0]
    dis = x.reshape(n, -1) - x
    indices = np.triu_indices(n, k=1)
    r= dis[indices[0], indices[1]]
    r.compute_gradients()
    return r


def surrogate_tau(a, b):
    da, db = pd(a), pd(b)
    return np.sum(sigm(da * db))


def lossf(predicted_D, expected_D, i=None, running=None):
    n = predicted_D.shape[0]
    r = np.array([0.])
    for pred, target in zip(predicted_D, expected_D):
        r += surrogate_tau(pred, target)
    return r / n


print(lossf(np.array([[5, 4, 3, 2, 1, 0]]), np.array([[0, 1, 2, 3, 4, 5]])))
print()
print(egrad(lossf)(np.array([[5, 40, 30, -2, 1, 0]]), np.array([[0, 1, 2, 3, 4, 5]])))
