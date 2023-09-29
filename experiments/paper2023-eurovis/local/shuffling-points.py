import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, vstack
from numpy.random import normal, default_rng
from sklearn.manifold import trustworthiness

from sortedness import rsortedness, sortedness, pwsortedness, global_pwsortedness
from sortedness.local import stress

# from sortedness.trustworthiness import trustworthiness

print("Intended to show how measures behave with increasing shuffling.")
rng = default_rng(seed=0)

xmax, ymax, n = 100, 100, 400
levels = [0, 3.125 / 2, 3.125, (3.125 + 6.25) / 2, 6.25, (6.25 + 12.5) / 2, 12.5, (12.5 + 25) / 2, 25, (25 + 50) / 2, 50, 62.5, 75, 87.5, 100]
k = 5


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, n_neighbors=k))


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
}

xlabel = "Shuffling Level (\\%)"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, xmax, n)
y = rnd.uniform(0, ymax, n)
X = vstack((x, y)).T

xmin = min(X[:, 0])
xmax = max(X[:, 0])
ymin = min(X[:, 1])
ymax = max(X[:, 1])
indices = rng.choice(len(X), size=len(X), replace=False)
replacement = np.random.rand(len(indices), 2)
replacement[:, 0] = xmin + replacement[:, 0] * (xmax - xmin)
replacement[:, 1] = ymin + replacement[:, 1] * (ymax - ymin)


def randomize_projection(M, pct):
    get = int((len(M) * pct) // 100)
    projection_rnd = X.copy()
    projection_rnd[indices[:get], :] = replacement[:get, :]
    return projection_rnd


d = {xlabel: levels}
for m, f in measures.items():
    print(m)
    d[m] = []
    for level in d[xlabel]:
        print(level, end=" ")
        X_ = randomize_projection(X, level)
        d[m].append(f(X, X_))
    print(d)

print("---------------------_")
_, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim([-25, 105])
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
    df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=False, fontsize=plt.rcParams["font.size"])

plt.grid()

plt.legend(loc=3)
plt.ylabel("")
plt.subplots_adjust(left=0.07, bottom=0.14, right=0.995, top=0.99)
# arq = '/home/davi/git/articles/sortedness/images/shuffling.pgf'
# plt.savefig(arq, bbox_inches='tight')
# with open(arq, "r") as f:
#     txt = f.read().replace("sffamily", "rmfamily")
# with open(arq, "w") as f:
#     f.write(txt)
plt.show()
