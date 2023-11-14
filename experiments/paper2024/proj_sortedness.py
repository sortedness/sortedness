import os
from pathlib import Path
from sys import argv

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
swap = "X_" if "swap" in argv else "X"
if "both" in argv:
    swap = "both"
datasets = [f"{d}3" for d in datasets]
onlyshowbest = "best" in argv
epochs = int(argv[1])
max_evals = int(argv[2])
with (sopen(schedule_uri) as db, sopen(remote_cache_uri) as remote):
    tasks = datasets if onlyshowbest else (Scheduler(db, timeout=60, mark_as_done=False) << datasets)
    for d in tasks:
        dataset_name = d[:-1]
        key = f"{dataset_name}a-trials"
        trials = remote[key] if key in remote else None

        if onlyshowbest:
            if trials is None:
                print(f"{dataset_name[:8]:8} has no trials yet")
                continue
            dct = {k: round(v[0], 3) if round(v[0]) - v[0] != 0 else int(v[0])
                   for k, v in trials.best_trial["misc"]["vals"].items()}
            print(f"{dataset_name[:8]:8} best:", dct, f"\tλ: {-trials.best_trial['result']['loss']:2.3f}\t", len(trials.results), flush=True)
            continue

        print(d, "---------------------------------------------------------------------------")
        kwargs = {} if trials is None else {"trials": trials}

        X, y = load_dataset(dataset_name)
        if len(trials.results) >= max_evals:
            print(f"Trials {len(trials.results)} >= {max_evals}")
            continue
        for max_evals_ in range(len(trials.results), max_evals + 1):
            X_, trials = balanced_embedding__opt(X, orderby=swap, epochs=epochs, max_evals=max_evals_, max_neurons=1000, recyclable=False, progressbar=True, show_parameters=True, return_trials=True, **kwargs)
            remote[key] = trials

        if X_.shape[0] != X.shape[0]:
            print('----------------------------------------------------')
            print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
            print('----------------------------------------------------')

        X_.tofile('proj_X_%s_SORT.csv' % dataset_name, sep=',')
