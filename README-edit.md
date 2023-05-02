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

<<sortedness>>

<<pwsortedness>>


** Copyright (c) 2022. Davi Pereira dos Santos and Tacito Neves**





## Grants
