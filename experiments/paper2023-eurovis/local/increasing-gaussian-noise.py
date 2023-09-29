from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lange import ap
from numpy import mean, vstack
from numpy.random import normal
from sklearn.manifold import trustworthiness

from sortedness import rsortedness, sortedness, pwsortedness, global_pwsortedness
from sortedness.local import stress
from sortedness.trustworthiness import trustworthiness

print("Intended to show sensitivity from irrelevant distortion to changing order.")
n = 17
k = 5
xmax = 100
ymax = 100
steps = 1000
stdev = 0.001


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return mean(trustworthiness(X, X_, k))


measures = {
    "$T_5$~~~~~~~~trustworthiness": tw,
    "$\\overline{\\lambda}_{\\tau_w}$~~~~~~reciprocal": lambda X, X_: mean(rsortedness(X, X_)),
    "$\\lambda_{\\tau_w}$~~~~~~sortedness": lambda X, X_: mean(sortedness(X, X_)),
    "$\\Lambda_{\\tau_w}$~~~~~pairwise": lambda X, X_: mean(pwsortedness(X, X_)),
    "$\\Lambda_{\\tau_1}$~~~~~~pairwise (global)": lambda X, X_: global_pwsortedness(X, X_)[0],
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - mean(stress(X, X_)),
}

xlabel = "Noise Amplitude"
rnd = np.random.default_rng(4)
x = rnd.uniform(0, xmax, n)
y = rnd.uniform(0, ymax, n)
X = vstack((x, y)).T
D = np.clip(X - rnd.normal(X, stdev), -stdev, stdev)
pprint(D)

d = {xlabel: [x * stdev for x in ap[1, 2, ..., steps]]}
for m, f in measures.items():
    print(m)
    d[m] = []
    for i in range(len(d[xlabel])):
        print(i, end=" ")
        X_ = X + i * D
        # pprint(X_)
        d[m].append(f(X, X_))
    print(d)

print("---------------------_")
_, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim([0.009, 0.602])
ax.set_ylim([0.995, 1.0001])
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
]):
    print("\n" + ylabel)
    df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=True, fontsize=plt.rcParams["font.size"])

plt.grid()

plt.legend(loc=3)
plt.ylabel("")
plt.subplots_adjust(left=0.07, bottom=0.14, right=0.995, top=0.99)
# arq = '/home/davi/git/articles/sortedness/images/gaussian-noise.pgf'
# plt.savefig(arq, bbox_inches='tight')
# with open(arq, "r") as f:
#     txt = f.read().replace("sffamily", "rmfamily")
# with open(arq, "w") as f:
#     f.write(txt)
plt.show()
