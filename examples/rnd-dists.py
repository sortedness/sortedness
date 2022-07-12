from random import shuffle

from src.robustress.rank import rdist_by_index_lw

max = -1
for l in range(1000):
    print(l, f"last max:{max}")
    max = 0
    for _ in range(1000):
        lst = list(range(l))
        shuffle(lst)
        d = rdist_by_index_lw(lst, normalized=False)
        dnorm = rdist_by_index_lw(lst)
        if dnorm > max:
            max = dnorm

        if dnorm > 1:
            print(dnorm)
            raise Exception("bug")
