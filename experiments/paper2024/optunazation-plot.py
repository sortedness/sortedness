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
from sys import argv

import optuna
from argvsucks import handle_command_line

from sortedness.config import optuna_uri
from sortedness.misc.dataset import load_dataset

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

floats = {"smoothness_tau": (0.00001, 100), "lr": (0.000001, 1), "alpha": (0.0000001, 0.99999), "weight_decay": (0.0000001, 0.99999), "momentum": (0.000001, 1)}
ints = {"epochs": (1, 400), "neurons": (2, 100), "batch_size": (1, 100)}
bools = {"centered": [False, True]}

dct = dict(
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

for dataset in datasets:
    study1 = optuna.load_study(storage=storage, study_name=f"{dataset}_alpha1_beta05_gamma1_d2_epoch_layers_optim_k_kg")
    study4 = optuna.load_study(storage=storage, study_name=f"{dataset}_alpha1_beta05_gamma4_d2_epoch_layers_optim_k_kg")

    for study in [study1, study4]:
        best = study.best_trial
        # if best.value > dct[dataset]:
        #     break
        print(f"{dataset:13} {str(load_dataset(dataset)[0].shape):13}", end="")
        df = study.trials_dataframe()
        mark = f"{100 * best.value / dct[dataset] - 100:03.3f}%" if best.value > dct[dataset] else "      "
        state = "COMPLETE"
        print(f"{best.value:03.10f} {mark} {df[df['state'] != state].shape[0]:3}/{df.shape[0]} not {state}  {best.params}")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    # plot_contour(study, params=["epoch", "hidden_layers"], target_name=dataset).show()
    # plot_param_importances(study, target_name=dataset).show()
    # plot_optimization_history(study, error_bar=True, target_name=dataset).show()
    # vis.plot_contour(study, params=["epoch", "smoothness_tau"], target_name=d).show()
    # exit()
    # vis.plot_contour(study, params=["smoothness_tau", "centered"]).show()
    # plot_slice(study, params=["hidden_layers"], target_name=dataset).show()
    # vis.plot_slice(study, params=["neurons", "batch_size"]).show()
    # vis.plot_parallel_coordinate(study, target_name=d).show()

    # exit()
