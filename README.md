![test](https://github.com/sortedness/sortedness/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/sortedness/sortedness/branch/main/graph/badge.svg)](https://codecov.io/gh/sortedness/sortedness)
<a href="https://pypi.org/project/sortedness">
<img src="https://img.shields.io/github/v/release/sortedness/sortedness?display_name=tag&sort=semver&color=blue" alt="github">
</a>
![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)
[![license: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2109.06028-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2109.06028) --->
[![API documentation](https://img.shields.io/badge/doc-API%20%28auto%29-a0a0a0.svg)](https://sortedness.github.io/sortedness)
[![DOI](https://zenodo.org/badge/513273889.svg)](https://zenodo.org/badge/latestdoi/513273889)
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

from sortedness import sortedness

# Some synthetic data.
mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

# Print `min`, `mean`, and `max` values.
s = sortedness(original, original)
print(min(s), sum(s) / len(s), max(s))
"""
1.0 1.0 1.0
"""
```

```python3

s = sortedness(original, projected2)
print(min(s), sum(s) / len(s), max(s))
"""
1.0 1.0 1.0
"""
```

```python3

s = sortedness(original, projected1)
print(min(s), sum(s) / len(s), max(s))
"""
0.432937128932 0.7813889452999166 0.944810120534
"""
```

```python3

s = sortedness(original, projectedrnd)
print(min(s), sum(s) / len(s), max(s))
"""
-0.578096068617 -0.06328160775358334 0.396112816715
"""
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

from sortedness import pwsortedness

# Some synthetic data.
mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

# Print `min`, `mean`, and `max` values.
s = pwsortedness(original, original)
print(min(s), sum(s) / len(s), max(s))
"""
1.0 1.0 1.0
"""
```

```python3

s = pwsortedness(original, projected2)
print(min(s), sum(s) / len(s), max(s))
"""
1.0 1.0 1.0
"""
```

```python3

s = pwsortedness(original, projected1)
print(min(s), sum(s) / len(s), max(s))
"""
0.730078995423 0.7744573488776667 0.837310352695
"""
```

```python3

s = pwsortedness(original, projectedrnd)
print(min(s), sum(s) / len(s), max(s))
"""
-0.198780473657 -0.0645984203715 0.147224384381
"""
```


</p>
</details>

**Sortedness**
<details>
<p>

```python3

import numpy as np
from numpy.random import permutation
from sklearn.decomposition import PCA

from sortedness import global_pwsortedness

# Some synthetic data.
mean = (1, 2)
cov = np.eye(2)
rng = np.random.default_rng(seed=0)
original = rng.multivariate_normal(mean, cov, size=12)
projected2 = PCA(n_components=2).fit_transform(original)
projected1 = PCA(n_components=1).fit_transform(original)
np.random.seed(0)
projectedrnd = permutation(original)

# Print measurement result and p-value.
s = global_pwsortedness(original, original)
print(list(s))
"""
[1.0, 3.6741408919675163e-93]
"""
```

```python3

s = global_pwsortedness(original, projected2)
print(list(s))
"""
[1.0, 3.6741408919675163e-93]
"""
```

```python3

s = global_pwsortedness(original, projected1)
print(list(s))
"""
[0.7715617715617715, 5.240847664048334e-20]
"""
```

```python3

s = global_pwsortedness(original, projectedrnd)
print(list(s))
"""
[-0.06107226107226107, 0.46847188611226276]
"""
```


</p>
</details>


** Copyright (c) 2022. Davi Pereira dos Santos and Tacito Neves**





## Grants
