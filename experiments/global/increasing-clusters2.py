import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import gp
from numpy import mean, vstack
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from sortedness.probabilistic import locglo

k = 5
limit = 2_000_000
distance, std = 100000, 1


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, n_neighbors=k))


measures = {
    # "$\\lambda_{\\tau_1}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_, f=kendalltau)),
    "$\\lambda_{\\tau_LG}$~~~~~~sortedness": lambda X, X_, n: mean(locglo(X, X_)),
    # "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_, n: mean(sortedness(X, X_)),
    # "$\\lambda_{\\tau_w\\%}$~~~~~~sortedness": lambda X, X_, n: mean(sortedness(X, X_, 1, weigher=lambda r: (1+r) / (1 + r) / n)),
    # "$\\lambda_{\\tau_G}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_, weigher=gaussian)),
    # "$\\lambda_{\\tau_w+G}$~~~~~~sortedness": lambda X, X_: mean((sortedness(X, X_) + sortedness(X, X_, weigher=gaussian)) / 2),
    # "$\\lambda_{\\tau_1+G}$~~~~~~sortedness": lambda X, X_: mean((sortedness(X, X_, f=kendalltau) + sortedness(X, X_, weigher=gaussian)) / 2),
    # "$\\lambda_{\\tau_1+w}$~~~~~~sortedness": lambda X, X_: mean((sortedness(X, X_, f=kendalltau) + sortedness(X, X_)) / 2),
}

xlabel = "Cluster Size"
d = {xlabel: [int(x) for x in gp[10, 30, ..., limit]]}
a, b, c = (-distance, 0), (0, 0), (distance, 0)
xlst = []
for center in [a, b, c]:
    Xi, _ = make_blobs(n_samples=limit, cluster_std=[std], centers=[center], n_features=2, random_state=1, shuffle=False)
    xlst.append(Xi)
for m, f in measures.items():
    print(m)
    d[m] = []
    for n in d[xlabel]:
        X = np.empty((3 * n, 2), dtype=float)
        for i in range(3):
            X[i * n:i * n + n] = xlst[i][:n]
        dx = np.array([[distance, 0]])
        X_ = vstack((X[:n], X[n:2 * n] + dx, X[2 * n:] - dx))
        s=f(X, X_, len(X) - 1)
        d[m].append(s)
        print("cluster size:", n, s)

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