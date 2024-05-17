from numpy import array


def merge(x, tmp, start, mid, end):
    k = i = start
    j = mid
    if x[mid - 1] <= x[mid]:
        return x
    while i < mid and j < end:
        if x[i] <= x[j]:
            tmp[k] = x[i]
            i = i + 1
        else:
            tmp[k] = x[j]
            j = j + 1
        k = k + 1
    if i < mid:
        x[k:end] = x[i:mid]
    x[start:k] = tmp[start:k]


def mergesort(x: array):
    n = len(x)
    if n < 2:
        return x
    tmp = x.copy()
    wid = 1
    while wid < n:
        for start in range(0, n, 2 * wid):
            mid = start + wid
            end = min(mid + wid, n)
            if mid < n:
                merge(x, tmp, start, mid, end)
        wid *= 2
