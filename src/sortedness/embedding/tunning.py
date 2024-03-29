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
from datetime import datetime
from functools import partial
from itertools import chain

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.exceptions import AllTrialsFailed
from numpy import mean
from numpy.random import default_rng
from scipy.stats import weightedtau, kendalltau
from torch.optim import RMSprop

from sortedness import sortedness
from sortedness.embedding.sortedness_ import balanced_embedding
from sortedness.embedding.surrogate import cau
from sortedness.local import geomean_np


def tuple2hyperopt(key, v):
    if type(v) is tuple and type(v[0]) is int:
        return hp.quniform(key, *v, 1)
    if type(v) is tuple and type(v[0]) is float:
        return hp.uniform(key, *v)
    return hp.choice(key, v)


# todo: replace max_smooth by max_lambda
def balanced_embedding__opt(X, d=2, gamma=4, k=17, K: int = "sqrt", alpha=0.5, beta=0.5, epochs=10,
                            max_neurons=100, max_smooth=2, max_batch=200,
                            embedding__param_space=None,
                            embedding_optimizer=RMSprop, embedding_optimizer__param_space=None,
                            hyperoptimizer_algorithm=None, max_evals=10, recyclable=True, progressbar=False,
                            min_K=100, max_K=1000, seed=0, track_best_model=True, return_only_X_=True, gpu=False, show_parameters=True, **hyperoptimizer_kwargs):
    """
    Warning: parameter `alpha` for balancing sortedness has nothing to do with embedding optimizer's `alpha`.

    Parameters
    ----------
    X
    d
    gamma
    k
    K
    alpha
    beta
    epochs
    max_neurons
    max_smooth
    max_batch
    embedding__param_space
    embedding_optimizer
    embedding_optimizer__param_space
    hyperoptimizer_algorithm
    max_evals
    recyclable
    progressbar
    min_K
    max_K
    seed
    track_best_model
    return_only_X_
    gpu
    show_parameters
    hyperoptimizer_kwargs

    Returns
    -------

    """
    if hyperoptimizer_algorithm is None:
        hyperoptimizer_algorithm = partial(tpe.suggest, n_startup_jobs=3, n_EI_candidates=6)
    if embedding__param_space is None:
        embedding__param_space = {}
    if embedding_optimizer__param_space is None:
        embedding_optimizer__param_space = {}

    for key, v in {"lambd": (0.0001, max_smooth), "neurons": (d, max_neurons), "batch_size": (1, min(max_batch, len(X)))}.items():
        if key not in embedding__param_space:
            embedding__param_space[key] = v
    for key, v in {"lr": (0.0001, 0.1), "alpha": (0.90, 0.99), "weight_decay": (0.0, 0.1), "momentum": (0.0, 0.1), "centered": [True, False]}.items():
        if key not in embedding_optimizer__param_space:
            embedding_optimizer__param_space[key] = v

    # Useful for recycling trials. Different settings should be reflected in the search space.
    if recyclable:
        fixed_space = dict(d=(float(d), d + 0.000001), alpha_embedding=(float(alpha), alpha + 0.000001),
                           gamma=(float(gamma), gamma + 0.000001), k=(float(k), k + 0.000001), beta=(float(beta), beta + 0.000001),
                           epochs=(float(epochs), epochs + 0.000001))
        if isinstance(K, int):
            fixed_space["K"] = (float(K), K + 0.000001)
    else:
        fixed_space = {}
    space = {k: tuple2hyperopt(k, v) for k, v in fixed_space.items()}

    for key, v in chain(embedding__param_space.items(), embedding_optimizer__param_space.items()):
        space[key] = tuple2hyperopt(key, v)

    if "algo" not in hyperoptimizer_kwargs:
        hyperoptimizer_kwargs["algo"] = hyperoptimizer_algorithm
    if "max_evals" not in hyperoptimizer_kwargs:
        hyperoptimizer_kwargs["max_evals"] = max_evals
    if "show_progressbar" not in hyperoptimizer_kwargs:
        hyperoptimizer_kwargs["show_progressbar"] = progressbar
    if "trials" not in hyperoptimizer_kwargs:
        trials = Trials()
        hyperoptimizer_kwargs["trials"] = trials
    else:
        trials: Trials = hyperoptimizer_kwargs["trials"]

    if len(trials) > 0:
        best = trials.best_trial
        result = best["result"]
        best_loss = [result["loss"]]
        if show_parameters:
            print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ", flush=True)
            dct1 = {k: round(v[0], 8) for k, v in best["misc"]["vals"].items()}
            print(f"{str(datetime.now())[:19]} {len(trials)}/{max_evals} Best:", dct1, f"λ:\t{-best['result']['loss']}", flush=True, sep="\t")
            # print(">>>>>>>>>>>>>>>>", quality, "<<<<<<<<<<<<<<<<<")
            # X_.tofile(f"proj-X_-{d}-SORT-quality_{quality}.csv", sep=',')
    else:
        best_loss = [-1]

    raise Exception(f"trocar para gaussian")
    def taus(r, r_):
        tau_local = weightedtau(r, r_, weigher=partial(cau, gamma), rank=False)[0]
        tau_global = kendalltau(r, r_)[0]
        return geomean_np(tau_local, tau_global)

    def objective(space):  # todo: replace lambd by lambda
        embedding__kwargs = {key: (v if key == "lambd" else int(v))
                             for key, v in space.items()
                             if key in ["lambd", "neurons", "batch_size"]}
        embedding_optimizer__kwargs = {key: v
                                       for key, v in space.items()
                                       if key not in chain(embedding__kwargs, fixed_space)}
        if "centered" in embedding_optimizer__kwargs:
            del embedding_optimizer__kwargs["centered"]
        if "alpha" in embedding_optimizer__kwargs:
            embedding_optimizer__kwargs["alpha_"] = embedding_optimizer__kwargs.pop("alpha")

        if show_parameters:
            print("·", flush=True)
            print(f"{str(datetime.now())[:19]}", flush=True)
            print(fixed_space, flush=True)
            print(embedding__kwargs, flush=True)
            print(embedding_optimizer__kwargs, flush=True)
            print("·", flush=True)
        tup = balanced_embedding(X, d, gamma, k, K, alpha, beta, epochs=epochs, **embedding__kwargs, embedding_optimizer=embedding_optimizer,
                                 min_K=min_K, max_K=max_K, seed=seed,
                                 track_best_model=track_best_model, return_only_X_=return_only_X_, gpu=gpu, **embedding_optimizer__kwargs)

        dct = {"status": STATUS_OK}
        if return_only_X_:
            dct["X_"] = tup
        else:
            dct["X_"], dct["model"], dct["quality"] = tup
        X_ = dct["X_"]

        if 0 < alpha < 1:
            loss = -mean(sortedness(X, X_, symmetric=True, f=taus))  # todo: replace symmetric by alpha
        elif alpha == 0:
            loss = -mean(sortedness(X, X_, symmetric=False, f=taus))
        elif alpha == 1:
            loss = -mean(sortedness(X_, X, symmetric=False, f=taus))
        else:
            raise Exception(f"Outside valid range: {alpha=}")
        dct["loss"] = loss

        if loss < best_loss[0]:
            best_loss[0] = loss
            if show_parameters:
                print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ", flush=True)
                try:
                    print(f"{str(datetime.now())[:19]} {len(trials) + 1}/{max_evals} Best: see above", f"λ:\t{loss} vs {loss}", flush=True, sep="\t")
                except AllTrialsFailed as e:
                    print(e)
        else:
            # Erase worse contents to save space.
            dct["X_"] = dct["model"] = None

        if show_parameters:
            print("\n", loss, flush=True)
            print("_", flush=True)

        return dct

    rnd = default_rng(seed)
    fmin(fn=objective, space=space, rstate=rnd, **hyperoptimizer_kwargs)
    best = trials.best_trial
    if show_parameters:
        print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ", flush=True)
        dct = {k: round(v[0], 3) for k, v in best["misc"]["vals"].items()}
        print(f"{str(datetime.now())[:19]} {len(trials)}/{max_evals} Best:", dct, f"λ:\t{-best['result']['loss']}", flush=True, sep="\t")

    result = best["result"]
    if return_only_X_:
        return result["X_"]
    if "model" in result:
        return result["X_"], result["model"], -result["loss"], trials
    return result["X_"], None, None, trials
