import os
from pathlib import Path
from pprint import pprint
from sys import argv

import numpy as np
from argvsucks import handle_command_line
from shelchemy import sopen
from shelchemy.scheduler import Scheduler

from sortedness.config import schedule_uri, remote_cache_uri
from sortedness.embedding.tunning import balanced_embedding__opt

"""
bash run:
for i in $(seq 2 100); do echo "++++++++++ $i epochs"; poetry run python experiments/paper2024/proj_sortedness.py $i $i alpha 0.5; done
"""


def load_dataset(dataset_name):
    data_dir = os.path.join(f"{Path.home()}/csv_proj_sortedness_out", dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X, y


print("Usage: proj_sortedness.py datasets=bank,cifar10,cnae9,coil20,epileptic,fashion_mnist,fmd,har,hatespeech,hiva,imdb,orl,secom,seismic,sentiment,sms,spambase,svhn [max_evals=10] [epochs=2] [alpha=0.5] [best]")
print("--------------------------------------------------------------------")
dct = handle_command_line(argv, datasets=list, max_evals=10, epochs=2, alpha=0.5, best=False)
datasets = dct["datasets"]
max_evals = dct["max_evals"]
alpha = dct["alpha"]
onlyshowbest = dct["best"]
epochs = dct["epochs"]
print()
print()
pprint(dct)
print()
print()

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
            X_, model, quality, trials = balanced_embedding__opt(X, alpha=alpha, epochs=epochs, max_evals=max_evals_, max_neurons=100, max_batch=50,
                                                                 embedding_optimizer__param_space={
                                                                     "alpha": (0.950, 0.999), "weight_decay": (0.00, 0.01), "momentum": (0.00, 0.01), "centered": [True, False]
                                                                 },
                                                                 track_best_model=True,
                                                                 progressbar=False, return_only_X_=False, show_parameters=True, **kwargs)
            remote[key] = trials

            if X_.shape[0] != X.shape[0]:
                print('----------------------------------------------------')
                print("Error running: Projection returned %d rows when %d rows were expected" % (X_.shape[0], X.shape[0]))
                print('----------------------------------------------------')

            print(">>>>>>>>>>>>>>>>", quality, "<<<<<<<<<<<<<<<<<")
            X_.tofile(f"proj-X_-{d}-SORT-quality_{quality}.csv", sep=',')
