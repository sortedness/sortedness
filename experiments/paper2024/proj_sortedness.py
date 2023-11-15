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


if len(argv) < 3:
    print("Usage: proj_sortedness.py epochs max_evals [alpha 0.7] [best]")
    exit()

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
alpha = float(argv[argv.index("alpha") + 1]) if "alpha" in argv else 0.5
onlyshowbest = "best" in argv
epochs = int(argv[1])
max_evals = int(argv[2])
with (sopen(schedule_uri) as db, sopen(remote_cache_uri) as remote):
    tasks = datasets if onlyshowbest else (Scheduler(db, timeout=60, mark_as_done=False) << datasets)
    for d in tasks:
        key = f"{d} - new trials"
        trials = remote[key] if key in remote else None

        if onlyshowbest:
            if trials is None:
                print(f"{d[:8]:8} has no trials yet")
                continue
            dct = {f"{k}": f"{v[0]:4.4f}" if round(v[0]) - v[0] != 0 else f"{int(v[0]):3}"
                   for k, v in trials.best_trial["misc"]["vals"].items()}
            print(f"{d[:4]:4} {-trials.best_trial['result']['loss']:4.3f} {len(trials.results)} {dct}", flush=True)
            # dct = {f"{k[:4]:4}": f"{v[0]:4.3f}" if round(v[0]) - v[0] != 0 else f"{int(v[0]):3}"
            #        for k, v in trials.best_trial["misc"]["vals"].items()}
            # print(f"{d[:4]:4} {-trials.best_trial['result']['loss']:4.3f} {len(trials.results)}{dct}", flush=True)
            continue

        print("\n", d, "=====================================================================================================\n", flush=True)
        kwargs, ini = ({}, 1) if trials is None else ({"trials": trials}, len(trials.results))

        X, y = load_dataset(d)
        if ini > max_evals:
            print(f"Trials {len(trials.results)} > {max_evals}")
            continue
        for max_evals_ in range(ini, max_evals + 1):
            X_, trials = balanced_embedding__opt(X, alpha=alpha, epochs=epochs, max_evals=max_evals_, max_neurons=100, max_batch=50,
                                                 embedding_optimizer__param_space={"alpha": (0.950, 0.999), "weight_decay": (0.00, 0.01), "momentum": (0.00, 0.01), "centered": [True, False]},
                                                 progressbar=True, show_parameters=True, return_trials=True, **kwargs)
            remote[key] = trials

            if X_.shape[0] != X.shape[0]:
                print('----------------------------------------------------')
                print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
                print('----------------------------------------------------')

            X_.tofile('proj_X_%s_SORT.csv' % d, sep=',')
