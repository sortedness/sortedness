import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp
from numpy import mean, vstack
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from sortedness import rsortedness, sortedness, pwsortedness, global_pwsortedness
from sortedness.local import stress

k = 5
limit = 100
distance, std = 100000, 1


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, n_neighbors=k))


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    # "$\\overline{\\lambda}_{\\tau_w}'$~~~~~~sortedness'": lambda X, X_: mean(sortedness1(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    # "$\\lambda_{\\tau_G}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_, weigher=ff)),
    # "$\\lambda_{\\tau_!}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_, weigher=lambda r: limit / (1 + r))),
    # "$\\Lambda_{\\tau_G}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_, weigher=ff)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
    # "gtau": lambda X, X_: 1 - mean(global_gtau(X, X_)),
    # "sortedness": lambda X, X_: mean(sortedness(X, X_, f=kendalltau)),
    # "$1-\\sigma_nm$~~nonmetric stress": lambda X, X_: 1 - mean(stress(X, X_, metric=False)),
    # "-gsortedness_w": lambda X, X_: gsortedness(X, X_, weigher=lambda r: r + 1),
    # "rsortedness": lambda X, X_: mean(rsortedness(X, X_, f=kendalltau)),
    # "continuity": lambda X, X_: mean(continuity(X, X_, k=15)),
    # "ρ": lambda X, X_: mean(sortedness(X, X_, f=spearmanr)),
    # "sortedness_x²": lambda X, X_: mean(sortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)))),
    # "rρ": lambda X, X_: mean(rsortedness(X, X_, f=spearmanr)),
    # "rsortedness_x²": lambda X, X_: mean(rsortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2)))),
    # "gρ": lambda X, X_: gsortedness(X, X_, f=spearmanr),
    # "gsortedness_x²": lambda X, X_: gsortedness(X, X_, f=partial(weightedtau, weigher=lambda x: 1 / (1 + x ** 2))),
    # "gsortedness_w": lambda X, X_: gsortedness(X, X_),
}

xlabel = "Cluster Size"
d = {xlabel: [int(x) for x in gp[2, 2.35, ..., limit]]}
a, b, c = (-distance, 0), (0, 0), (distance, 0)
xlst = []
for center in [a, b, c]:
    Xi, _ = make_blobs(n_samples=limit, cluster_std=[std], centers=[center], n_features=2, random_state=1, shuffle=False)
    xlst.append(Xi)
for m, f in measures.items():
    print(m)
    d[m] = []
    for n in d[xlabel]:
        print(n)
        X = np.empty((3 * n, 2), dtype=float)
        for i in range(3):
            X[i * n:i * n + n] = xlst[i][:n]
        dx = np.array([[distance, 0]])
        X_ = vstack((X[:n], X[n:2 * n] + dx, X[2 * n:] - dx))
        d[m].append(f(X, X_))

print("---------------------_")
_, ax = plt.subplots(figsize=(14, 9))
ax.set_ylim([-0.4, 1.05])
df = pd.DataFrame(d)
df = df.set_index(xlabel)  # .plot()
# ax.set_title('Loss curve', fontsize=15)
plt.rcParams["font.size"] = 35
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(plt.rcParams["font.size"])
for (ylabel, data), (style, width, color) in zip(list(d.items())[1:], [
    ("dotted", 1.5, "blue"),
    ("dotted", 3, "orange"),
    ("dotted", 3, "black"),
    ("-.", 3, "red"),
    ("dashed", 3, "purple"),
    ("dashed", 1.5, "brown"),
    ("solid", 2.5, "brown"),
]):
    print("\n" + ylabel)
    df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=True, fontsize=plt.rcParams["font.size"])

plt.grid()
plt.legend(loc=4)
plt.subplots_adjust(left=0.07, bottom=0.14, right=0.995, top=0.99)
# arq = '/home/davi/git/articles/sortedness/images/cluster-sizes.pgf'
# plt.savefig(arq, bbox_inches='tight')
# with open(arq, "r") as f:
#     txt = f.read().replace("sffamily", "rmfamily")
# with open(arq, "w") as f:
#     f.write(txt)
plt.show()
