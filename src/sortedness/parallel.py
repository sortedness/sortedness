import numpy as np
import pathos.multiprocessing as mp
from scipy.stats import rankdata


# def rank_alongrow(X, method="average", step=10):
#     n, m = X.shape
#     if step >= n or step < 1:
#         step = 1
#
#     def f(j):
#         X[j:j + step] = rankdata(X[j:j + step], axis=1, method=method)astype
#
#     jobs = mp.ThreadingPool().imap(f, range(0, n, step))
#     list(jobs)
#     return X.astype(int) - 1
#
#
# def rank_alongcol(X, method="average", step=10):
#     n, m = X.shape
#     if step >= n or step < 1:
#         step = 1
#
#     def f(j):
#         X[:, j:j + step] = rankdata(X[:, j:j + step], axis=0, method=method)astype
#
#     jobs = mp.ThreadingPool().imap(f, range(0, n, step))
#     list(jobs)
#     return X.astype(int) - 1


def rank_alongcol(X, method="average", step=10, parallel=True, **parallel_kwargs):
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel else map
    n = len(X)
    if step > n or step < 1:  # pragma: no cover
        step = n
    it = (X[:, j : j + step] for j in range(0, n, step))
    jobs = tmap(lambda M: rankdata(M, axis=0, method=method), it)
    return np.hstack(list(jobs)).astype(int) - 1


def rank_alongrow(X, method="average", step=10, parallel=True, **parallel_kwargs):
    tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel else map
    n = len(X)
    if step > n or step < 1:  # pragma: no cover
        step = n
    it = (X[j : j + step] for j in range(0, n, step))
    jobs = tmap(lambda M: rankdata(M, axis=1, method=method), it)
    return np.vstack(list(jobs)).astype(int) - 1


# def argsort_alongrow(X, step=10, parallel=True, **parallel_kwargs):
#     tmap = mp.ThreadingPool(**parallel_kwargs).imap if parallel else map
#     n = len(X)
#     if step > n or step < 1:
#         step = n
#     it = (X[j:j + step] for j in range(0, n, step))
#     jobs = tmap(lambda M: argsort(M, axis=1, kind="stable"), it)
#     # jobs = tmap(lambda M: rankdata(M, axis=1, method=method), it)
#     return np.vstack(list(jobs))

# def pw_sqeucl(M, **parallel_kwargs):
#     n = len(M)
#     li = (M[i] for i in range(n) for j in range(i + 1, n))
#     lj = (M[j] for i in range(n) for j in range(i + 1, n))
#     jobs = mp.ThreadingPool(**parallel_kwargs).imap(lambda l, g: [sqeuclidean(a, b) for a, b in zip(l, g)], ichunks(li, 20, asgenerators=False), ichunks(lj, 20, asgenerators=False))
#     return np.array(list(chain(*jobs)))

None

# set_num_threads(16)


# @numba.jit(nopython=True, parallel=True, cache=True)
# def pw_sqeucl_nb(M, M_):
#     n = len(M)
#     m = (n ** 2 - n) // 2
#     scores = np.zeros(m)
#     scores_ = np.zeros(m)
#     for i in prange(n):
#         num = 2 * n * i - i ** 2 - i
#         for j in range(i + 1, n):
#             c = num // 2 + j - i - 1
#             sub = M[i] - M[j]
#             scores[c] = -np.dot(sub, sub)
#             sub_ = M_[i] - M_[j]
#             scores_[c] = -np.dot(sub_, sub_)
#     return scores, scores_
