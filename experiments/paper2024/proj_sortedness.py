import os
from pathlib import Path

import numpy as np
from shelchemy import sopen
from shelchemy.scheduler import Scheduler

from sortedness.config import schedule_uri
from sortedness.embedding.tunning import balanced_embedding__opt


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
with sopen(schedule_uri) as db:
    for d in Scheduler(db, timeout=50) << datasets:
        dataset_name = d

        print(d)

        X, y = load_dataset(dataset_name)

        learning_optimizer__param_space = {"epochs": (1, 30), "lr": (0.001, 0.050), "alpha": (0.95, 0.99), "weight_decay": (0.0, 0.01), "momentum": (0.0, 0.01), "centered": [True, False]}

        # X_ = balanced0(X, False, epochs=1)
        X_ = balanced_embedding__opt(X, symmetric=False, k=15, global_k=15, max_evals=30, progressbar=True, learning_optimizer__param_space=learning_optimizer__param_space)

        # Verify
        if X_.shape[0] != X.shape[0]:
            print('----------------------------------------------------')
            print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
            print('----------------------------------------------------')

        X_.tofile('proj_X_%s_SORT.csv' % dataset_name, sep=',')
