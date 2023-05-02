![test](https://github.com/sortedness/sortedness/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/sortedness/sortedness/branch/main/graph/badge.svg)](https://codecov.io/gh/sortedness/sortedness)
<a href="https://pypi.org/project/sortedness">
<img src="https://img.shields.io/github/v/release/sortedness/sortedness?display_name=tag&sort=semver&color=blue" alt="github">
</a>
![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)
[![license: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!--- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5501845.svg)](https://doi.org/10.5281/zenodo.5501845)
[![arXiv](https://img.shields.io/badge/arXiv-2109.06028-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2109.06028) --->
[![API documentation](https://img.shields.io/badge/doc-API%20%28auto%29-a0a0a0.svg)](https://sortedness.github.io/sortedness)
[![Downloads](https://static.pepy.tech/badge/sortedness)](https://pepy.tech/project/sortedness)


# sortedness

`sortedness` is a measure of quality of data transformation, often dimensionality reduction.
It is less sensitive to irrelevant distortions and return values in a more meaningful interval than Kruskal stress formula I.
<br>This [Python library](https://pypi.org/project/sortedness) / [code](https://github.com/sortedness/sortedness) provides a reference implementation for the functions presented [here (paper unavailable until publication)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Nonparametric+Dimensionality+Reduction+Quality+Assessment+based+on+Sortedness+of+Unrestricted+Neighborhood&btnG=).

## Overview
Local variants return a value for each provided point. The global variant returns a single value for all points.
Any local variant can be used as a global measure by taking the mean value.

Local variants: `sortedness(X, X_)`, `pwsortedness(X, X_)`, `rsortedness(X, X_)`.

Global variant: `global_sortedness(X, X_)`.

## Python installation
### from package through pip
```bash
# Set up a virtualenv. 
python3 -m venv venv
source venv/bin/activate

# Install from PyPI
pip install -U sortedness
```

### from source
```bash
git clone https://github.com/sortedness/sortedness
cd sortedness
poetry install
```


### Examples

**Sortedness**
<details>
<p>

```python3

import numpy as np
from numpy.random import permutation
from sklearn.decomposition import PCA

from sortedness.local import sortedness

mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

s = sortedness(original, original)
print(min(s), max(s), s)
"""
1.0 1.0 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
"""
```

```python3

s = sortedness(original, projected2)
print(min(s), max(s), s)
"""
1.0 1.0 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
"""
```

```python3

s = sortedness(original, projected1)
print(min(s), max(s), s)
"""
0.432937128932 0.944810120534 [0.43293713 0.53333015 0.88412753 0.94481012 0.81485109 0.81330052
 0.76691474 0.91169619 0.88998817 0.90102615 0.61372341 0.86996213]
"""
```

```python3

s = sortedness(original, projectedrnd)
```


</p>
</details>

**Pairwise sortedness**
<details>
<p>

```python3

import numpy as np
from numpy.random import permutation
from sklearn.decomposition import PCA

from sortedness.local import pwsortedness

mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

s = pwsortedness(original, original)
print(min(s), max(s), s)
"""
1.0 1.0 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
"""
```

```python3

s = pwsortedness(original, projected2)
print(min(s), max(s), s)
"""
1.0 1.0 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
"""
```

```python3

s = pwsortedness(original, projected1)
print(min(s), max(s), s)
"""
0.730078995423 0.837310352695 [0.75892647 0.730079   0.83496865 0.73161226 0.75376525 0.83301104
 0.76695755 0.74759156 0.81434161 0.74067221 0.74425225 0.83731035]
"""
```

```python3

s = pwsortedness(original, projectedrnd)
```

```python3
print(min(s), max(s), s)

"""
-0.198780473657 0.147224384381 [-0.19878047 -0.14125391  0.03276727 -0.092844   -0.0866695   0.14722438
 -0.07603536 -0.08916877 -0.1373848  -0.10933483 -0.07774488  0.05404383]
"""
```


</p>
</details>


** Copyright (c) 2022. Davi Pereira dos Santos and Tacito Neves**





## Grants
