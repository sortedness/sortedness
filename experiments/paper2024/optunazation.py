#  Copyright (c) 2023. Davi Pereira dos Santos
#  This file is part of the sortedness project.
#  Please respect the license - more about this in the section (*) below.
#
#  sortedness is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sortedness is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
#
#  (*) Removing authorship by any means, e.g. by distribution of derived
#  works or verbatim, obfuscated, compiled or rewritten versions of any
#  part of this work is illegal and it is unethical regarding the effort and
#  time spent here.
#
from math import ceil
from sys import argv

import numpy as np
import optuna
from argvsucks import handle_command_line
from lange import gp, ap
from optuna import Trial
from optuna._callbacks import RetryFailedTrialCallback
from shelchemy import sopen
from shelchemy.scheduler import Scheduler

from sortedness.balanced import balanced_kendalltau_gaussian
from sortedness.config import schedule_uri, optuna_uri
from sortedness.embedding.sortedness_ import balanced_embedding
from sortedness.local import sortedness
from sortedness.misc.dataset import load_dataset


def getbest(st):
    try:
        return st.best_trial
    except ValueError as e:
        if "Record does not exist" in str(e):
            return None


dct = handle_command_line(argv, datasets=list, max_epochs=10000, max_neurons_first_layer=1000, max_hidden_layers=10, K=100, kappa=5, alpha=0.5, beta=0.5, pct=90)
print("Usage: optunazation.py datasets=bank,cifar10,cnae9,coil20,epileptic,fashion_mnist,fmd,har,hatespeech,hiva,imdb,orl,secom,seismic,sentiment,sms,spambase,svhn [max_epochs=10000] [max_neurons_first_layer=1000] [max_hidden_layers=10] [K=100] [kappa=5] [alpha=0.5] [beta=0.5] [pct=90]")
print("--------------------------------------------------------------------")
for tup in dct.items():
    print(tup)
print("--------------------------------------------------------------------")
print()
datasets = dct["datasets"]
max_epochs = dct["max_epochs"]
K = dct["K"]
max_neurons_first_layer = dct["max_neurons_first_layer"]
max_hidden_layers = dct["max_hidden_layers"]
kappa = dct["kappa"]
alpha = dct["alpha"]
beta = dct["beta"]
pct = dct["pct"]

current_uri = optuna.storages.RDBStorage(url=optuna_uri, heartbeat_interval=60, grace_period=120, failed_trial_callback=RetryFailedTrialCallback())

with sopen(schedule_uri) as db:
    for epochsf in gp[3, 3.2, ..., max_epochs]:
        print(f"{int(epochsf)=}\t|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        tasks = [(d, round(epochsf, 1), K, f"{kappa=} {alpha=} {beta=} {pct=}") for d in datasets]
        for dataset, __, ___, txt in Scheduler(db, timeout=30) << tasks:
            name = f"{dataset}_{txt.replace('=', '_').replace(' ', '__')}"
            print(f"{name=} {epochsf=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            epochs = int(epochsf)
            study = optuna.create_study(storage=current_uri, study_name=name, load_if_exists=True, direction="maximize")

            # print("ADD NEW HYPERPARAMETERS !!!!!!")
            # # sleep(5)
            # translations = dict()
            # dist_def__s = dict(epochs=(IntDistribution(1, 100000), None))
            # recreate_study_with_more_hyperparameters(study, newuri, name, dist_def__s, translations)
            # print("++++++++++++++++++++++++++++++§§§§§§§§§§§§§§§§§§§§§§§§§§")
            # continue

            successful = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            if (best := getbest(study)) is not None:
                print(f"{successful}/{len(study.trials)}\tPrevious best λ:", best.value, "\tParams:", best.params)
            else:
                print("Starting from scratch ------------------------------------------------------")
            X, y = load_dataset(dataset)


            def objective(trial: Trial):
                trial.suggest_int("epochs", epochs, epochs)
                trial.suggest_int("K", K, K)

                # topology
                n_layers = trial.suggest_int("hidden_layers", 1, max_hidden_layers)
                layers, afs = [], []
                for i in ap[1, 2, ..., max_hidden_layers]:
                    if i <= n_layers:
                        neurons = trial.suggest_int(f"neurons{i}", 1, ceil(max_neurons_first_layer / i))
                        af = trial.suggest_categorical(f"actfunc{i}", ("tanh", "sigm", "relu", "relu6"))
                        afs.append(af)
                    else:
                        neurons = trial.suggest_int(f"neurons{i}", 0, 0)
                    layers.append(neurons)

                hyperoptimizer = trial.suggest_int("hyperoptimizer", 0, 2)
                if hyperoptimizer == 0:
                    res = balanced_embedding(
                        X, kappa=kappa, K=K, alpha=alpha, beta=beta, epochs=epochs,
                        lambd=trial.suggest_float("lambd", 0.00001, 100, log=True),
                        batch_size=trial.suggest_int("batch_size", 1, 100),
                        hidden_layers=layers, activation_functions=afs,
                        min_K=17, max_K=10000, pct=pct,
                        lr=trial.suggest_float("lr", 0.000001, 1, log=True),
                        momentum=trial.suggest_float("momentum", 0.000001, 1, log=True),
                        weight_decay=trial.suggest_float("weight_decay", 0.0000001, 1, log=True),
                        alpha_=1 - trial.suggest_float("alpha_", 0.0000001, 1, log=True),
                        centered=trial.suggest_categorical("centered", (False, True)),
                        return_only_X_=False, verbose=False)
                else:
                    res = balanced_embedding(
                        X, kappa=kappa, K=K, alpha=alpha, beta=beta, epochs=epochs,
                        lambd=trial.suggest_float("lambd", 0.00001, 100, log=True),
                        batch_size=trial.suggest_int("batch_size", 1, 100),
                        hidden_layers=layers, activation_functions=afs,
                        min_K=17, max_K=10000, pct=pct,
                        hyperoptimizer=hyperoptimizer,
                        sgd_alpha=trial.suggest_float("sgd_alpha", 0.000001, 1, log=True),
                        sgd_mu=trial.suggest_float("sgd_mu", 0.000001, 1, log=True),
                        return_only_X_=False, verbose=False)
                trial.suggest_int("epoch", res["epoch"], res["epoch"])
                print("Target measure...", flush=True)
                qualities = sortedness(X, res["X_"], symmetric=False, f=balanced_kendalltau_gaussian, alpha=alpha, beta=beta, kappa=kappa, pct=pct)
                print("\t\t\t\t\tOK", flush=True)
                quality = np.mean(qualities)
                if (best := getbest(study)) is None or quality > best.value:
                    res["X_"].tofile(f"optuna-{dataset.ljust(20, '_')}-best_X__{name}.csv", sep=',')
                    qualities.tofile(f"optuna-{dataset.ljust(20, '_')}-best_quality-for-each-point-{dataset}_{name}.csv", sep=',')
                    print("CSVs saved")
                return quality


            print(f"««««««« Retry FAILed trials «««««««")
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.FAIL:
                    print("Retrying params of trial:", trial.number)
                    study.enqueue_trial(trial.params, skip_if_exists=True)
            print("»»»»»»»»»»»»»»»»»»»»»»»»»»»»")

            study.optimize(objective, n_trials=12, n_jobs=1)
