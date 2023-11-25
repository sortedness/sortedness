import os
from pathlib import Path

import numpy as np

from sortedness.embedding.sortedness_ import balanced_embedding_tacito
from sortedness.local import balanced_kendalltau
from sortedness.local import sortedness


def load_dataset(dataset_name):
    data_dir = os.path.join(f"{Path.home()}/csv_proj_sortedness_out", dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X, y


datasets = [
    "bank",
    "cnae9",
    "epileptic",
    "fmd",
    "hatespeech",
    "imdb",
    "secom",
    "sentiment",
    "spambase",
    "cifar10",
    "coil20",
    "fashion_mnist",
    "har",
    "hiva",
    "orl",
    "seismic",
    "sms",
    "svhn"
]
for d in datasets:
    print("\n", d, "=====================================================================================================\n", flush=True)
    X, y = load_dataset(d)
    X = X[:100]
    X_, model, surrogate_quality = balanced_embedding_tacito(
        X,
        # Parâmetros interessantes de variar:
        d=2,  # dimensões;
        gamma=4,  # alcance de vizinhança da Cauchy;
        beta=0.5,  # balanço entre localidade e globalidade;
        smoothness_tau=1,  # suavidade da derivada para o gradiente descendente;
        neurons=2,  # número de neurônios da única camada oculta;
        epochs=2,  # número de épocas de treinamento (quantas vezes o dataset será apresentado à rede neural);
        seed=0,  # semente para o gerador pseudo-aleatório de pesos iniciais para as sinapses da rede.
        verbose=True
    )

    if X_.shape[0] != X.shape[0]:
        print('----------------------------------------------------')
        print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
        print('----------------------------------------------------')

    qualities = sortedness(X_, X, symmetric=False, f=balanced_kendalltau)
    quality = np.mean(qualities)

    print(">>>>>>>>>>>>>>>>", quality, "<<<<<<<<<<<<<<<<<")
    X_.tofile(f"proj-X_-{d}-SORT-quality_{quality}-changing-parameters.csv", sep=',')
