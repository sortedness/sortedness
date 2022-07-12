from numpy import eye
from numpy.linalg import norm
from numpy.random import randint, shuffle
from sympy.utilities.iterables import multiset_permutations

from src.robustress.rank import rank_dist__by_index

old = 0
for l in range(1, 10):
    lst = list(range(l))
    d = 0
    c = 0
    for p in multiset_permutations(lst):
        d += rank_dist__by_index(p, normalized=False)
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
