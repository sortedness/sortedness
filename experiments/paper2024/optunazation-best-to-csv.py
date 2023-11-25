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
from pathlib import Path
from sys import argv

import numpy as np
import optuna
from argvsucks import handle_command_line

from sortedness import sortedness
from sortedness.config import near_cache_uri
from sortedness.embedding import balanced_embedding
from sortedness.local import balanced_kendalltau


def load_dataset(dataset_name):
    data_dir = os.path.join(f"{Path.home()}/csv_proj_sortedness_out", dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X, y


dct = handle_command_line(argv, datasets=list)
print("Usage: optunazation.py datasets=bank,cifar10,cnae9,coil20,epileptic,fashion_mnist,fmd,har,hatespeech,hiva,imdb,orl,secom,seismic,sentiment,sms,spambase,svhn")
print("--------------------------------------------------------------------")
for tup in dct.items():
    print(tup)
print()
datasets = dct["datasets"]

suffix = "alpha1_beta05_gamma4_k17_d2_epochbest_new2addneurons2__emboptmultil"

floats = {"smoothness_tau": (0.00001, 100), "lr": (0.000001, 1), "alpha": (0.0000001, 0.99999), "weight_decay": (0.0000001, 0.99999), "momentum": (0.000001, 1)}
ints = {"epochs": (1, 400), "neurons": (2, 100), "batch_size": (1, 100)}
bools = {"centered": [False, True]}

for d in datasets:
    study = optuna.load_study(storage=near_cache_uri, study_name=f"{d}_{suffix}")
    best = study.best_trial
    print(f"{d:17} {best.value:03.3f} {best.params}")

    X, y = load_dataset(d)
    X_ = balanced_embedding(X, alpha=1, epochs=100, return_only_X_=True, **best.params)
    qualities = sortedness(X_, X, symmetric=False, f=balanced_kendalltau)
    quality = np.mean(qualities)
    print(">>>>>>>>>>>>>>>>", quality, "<<<<<<<<<<<<<<<<<")

    X_.tofile(f"/tmp/optuna-best_X_-points-{d}_{suffix}-quality_{best.value}.csv", sep=',')
    qualities.tofile(f"/tmp/optuna-best_quality-for-each-point-{d}_{suffix}-quality_{best.value}.csv", sep=',')
