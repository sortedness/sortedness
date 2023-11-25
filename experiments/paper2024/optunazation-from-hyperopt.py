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
import os
from collections import ChainMap
from pathlib import Path
from pprint import pprint
from sys import argv
from time import sleep

import numpy as np
import optuna
from argvsucks import handle_command_line
from optuna._callbacks import RetryFailedTrialCallback
from optuna.distributions import FloatDistribution, CategoricalDistribution, IntDistribution
from shelchemy import sopen

from sortedness.config import remote_cache_uri, near_cache_uri


def load_dataset(dataset_name):
    data_dir = os.path.join(f"{Path.home()}/csv_proj_sortedness_out", dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X, y


dct = handle_command_line(argv, datasets=list)
print("Usage: optunazation.py datasets=bank,cifar10,cnae9,coil20,epileptic,fashion_mnist,fmd,har,hatespeech,hiva,imdb,orl,secom,seismic,sentiment,sms,spambase,svhn")
print("--------------------------------------------------------------------")
print()
print()
for tup in dct.items():
    print(tup)
print()
print()
datasets = dct["datasets"]

suffix = "alpha1_beta05_gamma4_k17_d2"
print("deletar tudo? 5s...")
sleep(5)

storage = optuna.storages.RDBStorage(url=near_cache_uri, heartbeat_interval=60, grace_period=120, failed_trial_callback=RetryFailedTrialCallback())
with sopen(remote_cache_uri) as remote:
    for d in datasets:
        # if d!="hiva":
        #     continue
        print(d, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        try:
            optuna.delete_study(storage=near_cache_uri, study_name=f"{d}_{suffix}")
        except KeyError:
            pass

        print("importar tudo? 30s...")
        sleep(30)

        key = f"{d} - new trials"
        hyperopt_trials = remote[key]
        hyperopt_params = [{k: round(v[0], 5) for k, v in tr['misc']['vals'].items()} for tr in hyperopt_trials.trials]
        floats = {"smoothness_tau": (0.00001, 100), "lr": (0.000001, 1), "alpha": (0.0000001, 0.99999), "weight_decay": (0.0000001, 0.99999), "momentum": (0.000001, 1)}
        ints = {"epochs": (1, 400), "neurons": (2, 100), "batch_size": (1, 100)}
        bools = {"centered": [False, True]}
        study = optuna.create_study(storage=storage, study_name=f"{d}_{suffix}", direction='maximize')
        for result, params in zip(hyperopt_trials.results, hyperopt_params):
            if params["alpha_embedding"] == 0.5:
                continue
            del params["alpha_embedding"]
            del params["beta"]
            del params["gamma"]
            del params["k"]
            del params["d"]
            params["smoothness_tau"] = params.pop("smooothness_tau")
            dists_lst = [
                {k: FloatDistribution(*args, log=True) for k, args in floats.items()},
                {k: IntDistribution(*args) for k, args in ints.items()},
                {k: CategoricalDistribution(args) for k, args in bools.items()}
            ]
            for k, v in params.items():
                if v == 0 and k != "centered":
                    params[k] = 0.0000001
            trial = optuna.trial.create_trial(params=params, distributions=dict(ChainMap(*dists_lst)), value=-result['loss'])
            study.add_trial(trial)

        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        pprint(study.best_trial.params)
