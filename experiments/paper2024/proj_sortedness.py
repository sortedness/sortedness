import os
from pathlib import Path

import numpy as np
from shelchemy import sopen
from shelchemy.scheduler import Scheduler

from sortedness.config import schedule_uri, remote_cache_uri
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
with (sopen(schedule_uri) as db, sopen(remote_cache_uri) as remote):
    for d in Scheduler(db) << datasets:
        dataset_name = d
        X, y = load_dataset(dataset_name)
        kwargs = {"trials": remote[key]} if (key := f"{d}-trials") in remote else {}
        X_, trials = balanced_embedding__opt(X, symmetric=False, embedding__param_space={"epochs": (1, 30)}, max_evals=30, progressbar=True, show_parameters=True, return_trials=True, **kwargs)
        remote[key] = trials

        if X_.shape[0] != X.shape[0]:
            print('----------------------------------------------------')
            print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
            print('----------------------------------------------------')

        X_.tofile('proj_X_%s_SORT.csv' % dataset_name, sep=',')
