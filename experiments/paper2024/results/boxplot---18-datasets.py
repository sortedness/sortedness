import matplotlib.pyplot as plt
import optuna
import seaborn as sns
from pandas import DataFrame

from sortedness import sortedness
from sortedness.config import optuna_uri
from sortedness.local import stress, balanced_kendalltau_gaussian
from sortedness.misc.dataset import load_dataset
from sortedness.misc.trustworthiness import trustworthiness

datasets = dict(
    bank=0.4915,
    cifar10=0.6843,
    cnae9=0.6102,
    coil20=0.6927,
    epileptic=0.5144,
    fashion_mnist=0.7600,
    fmd=0.6067,
    har=0.8781,
    hatespeech=0.3740,
    hiva=0.599358,
    imdb=0.4556,
    orl=0.997,
    secom=0.600921684717123837,
    seismic=0.7301,
    sentiment=0.3420,
    sms=0.5688,
    spambase=0.6599,
    svhn=0.783
)
storage = optuna.storages.RDBStorage(url=optuna_uri)

k = 7


def tw(X, X_):
    if k >= len(X) / 2:
        return 0
    return trustworthiness(X, X_, k)


measures_dct = {
    f"$T_{k}$~~~~~~~~trustworthiness": tw,
    f"$T_{k}$~~~~~~~~continuity": lambda X, X_: tw(X_, X),
    # "$\\lambda_{\\lambda}$~~~~~~sortedness": lambda X, X_: sortedness(X_, X, symmetric=False),
    "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\beta=0,\\alpha=0$": lambda X, X_: sortedness(X, X_, symmetric=False, f=balanced_kendalltau_gaussian, alpha=0, beta=0),
    "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\beta=0,\\alpha=1$": lambda X, X_: sortedness(X, X_, symmetric=False, f=balanced_kendalltau_gaussian, alpha=1, beta=0),
    "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\beta=0.5,\\alpha=0$": lambda X, X_: sortedness(X, X_, symmetric=False, f=balanced_kendalltau_gaussian, alpha=0, beta=0.5),
    "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\beta=0.5,\\alpha=1$": lambda X, X_: sortedness(X, X_, symmetric=False, f=balanced_kendalltau_gaussian, alpha=1, beta=0.5),
    "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\beta=1$": lambda X, X_: sortedness(X, X_, symmetric=False, f=balanced_kendalltau_gaussian, beta=1),
    # "$\\Lambda_{\\lambda}$~~~~~~balanced sortedness $\\gamma=4$": lambda X, X_: sortedness(X_, X, symmetric=False, f=balanced_kendalltau, gamma=4),
    "$1-\\sigma_1$~~metric stress": lambda X, X_: 1 - stress(X, X_),
}

n = 1000
xlabel = "Datasets"
xlabels, measures, values = [], [], []
for m, f in measures_dct.items():
    print(m)
    for dataset in datasets:
        print(dataset, end=" ")
        X, y, X_ = load_dataset(dataset, True)
        X, X_ = X[:n], X_[:n]
        xlabels.extend([dataset] * len(X))
        measures.extend([m] * len(X))
        values.extend(f(X, X_))

print("---------------------_")
_, ax = plt.subplots(figsize=(14, 9))
ax.set_ylim([-0.1, 1.1])
df = DataFrame({xlabel: xlabels, "Measure": measures, "Value": values})
# ax.set_title('Loss curve', fontsize=15)
plt.rcParams["font.size"] = 10
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(plt.rcParams["font.size"])
    # linestyle=style, lw=width, color=color, logy=False, logx=False, fontsize=plt.rcParams["font.size"])

sns.boxplot(ax=ax, width=0.7, y='Value', x=xlabel, data=df, palette=["blue", "orange", "gray", "red", "brown", "purple"], hue='Measure')
plt.grid()
plt.legend(loc=3)
plt.ylabel("")
plt.subplots_adjust(left=0.07, bottom=0.14, right=0.995, top=0.99)
# arq = '/home/davi/git/articles/sortedness/images/boxplot.pgf'
# plt.savefig(arq, bbox_inches='tight')
# with open(arq, "r") as f:
#     txt = f.read().replace("sffamily", "rmfamily")
# with open(arq, "w") as f:
#     f.write(txt)
plt.show()
