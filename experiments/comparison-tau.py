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
from scipy.stats import kendalltau
from torch import tensor

from sortedness.new.quality.measure.pairwise import softtau

l = .0001
print(float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 3, 4]), lambd=l)), float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 2, 4]), lambd=l)))
print(kendalltau([1, 2, 2, 4], [1, 2, 3, 4], variant='b').statistic, kendalltau([1, 2, 2, 4], [1, 2, 2, 4], variant='b').statistic)
print()
l = 1000
print(float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 3, 4]), lambd=l)), float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 2, 4]), lambd=l)))
print(kendalltau([1, 2, 2, 4], [1, 2, 3, 4], variant='b').statistic, kendalltau([1, 2, 2, 4], [1, 2, 2, 4], variant='b').statistic)

print()
a = [2, 2, 2, 2]
b = [2, 2, 2, 2]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001)), float(softtau(tensor(a), tensor(b), lambd=l)))
print(kendalltau(a, b, variant='b').statistic)
a = [2, 2, 2, 3]
b = [2, 2, 2, 2]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001)), float(softtau(tensor(a), tensor(b), lambd=l)))
print(kendalltau(a, b, variant='b').statistic)
print()
print()
print()
a = [1, 2, 3, 3]
b = [2, 1, 3, 3]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001)))
print(kendalltau(a, b, variant='b').statistic)
print()
print()
a = [1, 2, 3, 4]
b = [2, 1, 3, 3]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001)))
print(kendalltau(a, b, variant='b').statistic)


print("-----------------------------------------")

l = .0001
print(float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 3, 4]), lambd=l, tau=False)), float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 2, 4]), lambd=l, tau=False)))
print(kendalltau([1, 2, 2, 4], [1, 2, 3, 4], variant='b').statistic, kendalltau([1, 2, 2, 4], [1, 2, 2, 4], variant='b').statistic)
print()
l = 1000
print(float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 3, 4]), lambd=l, tau=False)), float(softtau(tensor([1, 2, 2, 4]), tensor([1, 2, 2, 4]), lambd=l, tau=False)))
print(kendalltau([1, 2, 2, 4], [1, 2, 3, 4], variant='b').statistic, kendalltau([1, 2, 2, 4], [1, 2, 2, 4], variant='b').statistic)

print()
a = [2, 2, 2, 2]
b = [2, 2, 2, 2]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001, tau=False)), float(softtau(tensor(a), tensor(b), lambd=l, tau=False)))
print(kendalltau(a, b, variant='b').statistic)
a = [2, 2, 2, 3]
b = [2, 2, 2, 2]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001, tau=False)), float(softtau(tensor(a), tensor(b), lambd=l, tau=False)))
print(kendalltau(a, b, variant='b').statistic)
print()
print()
print()
a = [1, 2, 3, 3]
b = [2, 1, 3, 3]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001, tau=False)))
print(kendalltau(a, b, variant='b').statistic)
print()
print()
a = [1, 2, 3, 4]
b = [2, 1, 3, 3]
print(float(softtau(tensor(a), tensor(b), lambd=0.00001, tau=False)))
print(kendalltau(a, b, variant='b').statistic)
