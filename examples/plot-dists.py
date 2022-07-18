import matplotlib.pyplot as plt
from lange import ap
from sympy.utilities.iterables import multiset_permutations

from sortedness.rank import rdist_by_index_lw

ds = {}
for l in ap[5, 6, ..., 8]:
    print(l)
    lst = list(ap[0, 1, ..., l])
    ds[l] = []
    for p in multiset_permutations(lst):
        ds[l].append(rdist_by_index_lw(p, normalized=False))
        # ds[l].append(rank_based_dist__by_index(p, normalized=True))
    ds[l].sort()
    # plt.plot(ds[l])

plt.hist(ds.values(), bins=25, log=True, histtype="barstacked")

# plt.scatter(arr[:, 0], arr[:, 1], s=pen_width, c=colors)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
