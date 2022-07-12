from random import shuffle

from src.robustress.main import rank_dist__by_index

max = -1
for l in range(1000):
    print(l, f"last max:{max}")
    max = 0
    for _ in range(1000):
        lst = list(range(l))
        shuffle(lst)
        d = rank_dist__by_index(lst, normalized=False)
        dnorm = rank_dist__by_index(lst)
        if dnorm > max:
            max = dnorm

        if dnorm > 1:
            print(dnorm)
            raise Exception("bug")
