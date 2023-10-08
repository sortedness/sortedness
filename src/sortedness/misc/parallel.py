import numpy as np
import pathos.multiprocessing as mp
from scipy.stats import rankdata


def rank_alongcol(X, method="average", step=10, parallel=True, **parallel_kwargs):   # pragma: no cover
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel else map
    n = len(X)
    if step > n or step < 1:  # pragma: no cover
        step = n
    it = (X[:, j : j + step] for j in range(0, n, step))
    jobs = tmap(lambda M: rankdata(M, axis=0, method=method), it)
    return np.hstack(list(jobs)).round().astype(int) - 1


def rank_alongrow(X, method="average", step=10, parallel=True, **parallel_kwargs):
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel else map
    n = len(X)
    if step > n or step < 1:  # pragma: no cover
        step = n
    it = (X[j : j + step] for j in range(0, n, step))
    jobs = tmap(lambda M: rankdata(M, axis=1, method=method), it)
    return np.vstack(list(jobs)).round().astype(int) - 1
