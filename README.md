![test](https://github.com/davips/robustress/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/davips/robustress/branch/main/graph/badge.svg)](https://codecov.io/gh/davips/robustress)
<a href="https://pypi.org/project/robustress">
<img src="https://img.shields.io/pypi/v/robustress.svg?label=release&color=blue&style=flat-square" alt="pypi">
</a>
![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue.svg)
[![license: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5501845.svg)](https://doi.org/10.5281/zenodo.5501845)
[![arXiv](https://img.shields.io/badge/arXiv-2109.06028-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2109.06028)
[![API documentation](https://img.shields.io/badge/doc-API%20%28auto%29-a0a0a0.svg)](https://davips.github.io/robustress)


# robustress
 


`robustress` solves some issues with Kruskal stress.<br>This [Python library](https://pypi.org/project/robustress) / [code](https://github.com/davips/robustress) provides a reference implementation for the stress function presented [here](https://arxiv.org/abs/2109.06028.9999).

## Overview

## Python installation
### from package through pip
```bash
# Set up a virtualenv. 
python3 -m venv venv
source venv/bin/activate

# Install from PyPI
pip install robustress
```

### from source
```bash
git clone https://github.com/davips/robustress
cd robustress
poetry install
```

### Examples
Some usage examples.

**Just a placeholder for futures examples:**
<details>
<p>

```python3
from numpy import eye
from numpy.linalg import norm
from numpy.random import randint, shuffle
from sympy.utilities.iterables import multiset_permutations

from robustress.rank import rdist_by_index_lw

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
