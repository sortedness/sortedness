![test](https://github.com/sortedness/sortedness/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/sortedness/sortedness/branch/main/graph/badge.svg)](https://codecov.io/gh/sortedness/sortedness)
<a href="https://pypi.org/project/sortedness">
<img src="https://img.shields.io/pypi/v/sortedness.svg?label=release&color=blue&style=flat-square" alt="pypi">
</a>
![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)
[![license: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5501845.svg)](https://doi.org/10.5281/zenodo.5501845)
[![arXiv](https://img.shields.io/badge/arXiv-2109.06028-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2109.06028)
[![API documentation](https://img.shields.io/badge/doc-API%20%28auto%29-a0a0a0.svg)](https://sortedness.github.io/sortedness)


# sortedness
 


`sortedness` is a measure of how much distortion occurred after dimensionality reduction.
It complements Kruskal's stress.<br>This [Python library](https://pypi.org/project/sortedness) / [code](https://github.com/sortedness/sortedness) provides a reference implementation for the stress function presented [here](https://arxiv.org/abs/2109.06028.9999).

## Overview

## Python installation
### from package through pip
```bash
# Set up a virtualenv. 
python3 -m venv venv
source venv/bin/activate

# Install from PyPI
pip install sortedness
```

### from source
```bash
git clone https://github.com/sortedness/sortedness
cd sortedness
poetry install
```

### Examples
Some usage examples.

** Copyright (c) 2022. Davi Pereira dos Santos**
<details>
<p>

```python3
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

# Just a placeholder for futures examples:
from numpy import eye
from numpy.linalg import norm
from numpy.random import randint, shuffle
from sympy.utilities.iterables import multiset_permutations

from sortedness.rank import rdist_by_index_lw

old = 0
for l in range(1, 10):
    lst = list(range(l))
    d = 0
    c = 0
    for p in multiset_permutations(lst):
        d += rdist_by_index_lw(p, normalized=False)
        c += 1
    d /= c
    print(l, "\t", d, "\t", d - old)
    old = d
"""
normalized:
    ~0.67                                   convergent?
otherwise:
    1.1, 1.8, 2.5, 3.3, 4.1, 4.9, 5.7, ...  divergent
"""

"""
1 	 0.0 	 0.0
2 	 0.5 	 0.5
3 	 1.1111111111333334 	 0.6111111111333334
4 	 1.7916666666833334 	 0.68055555555
5 	 2.5200000000099996 	 0.7283333333266662
6 	 3.2833333333449786 	 0.763333333334979
7 	 4.07346938774276 	 0.7901360543977813
8 	 4.884821428570203 	 0.811352040827443
9 	 5.713403880078337 	 0.8285824515081339
"""
```


</p>
</details>



## Grants
