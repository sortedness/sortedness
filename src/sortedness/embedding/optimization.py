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

import optuna
from optuna import Study
from optuna.trial import TrialState


def recreate_study_with_more_hyperparameters(study: Study, storage, newname, dist_def__s: dict, translations: dict):
    newstudy = optuna.create_study(storage=storage, study_name=newname, direction=study.direction)
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue

        # Modify previous trials. These changes are not persisted to the storage.
        params = trial.params
        dists = trial.distributions

        for hp_name, (hp_distribution, default_value) in dist_def__s.items():
            if hp_name == "epochs":
                default_value = params["epoch"] + 1
            params[hp_name] = default_value
            dists[hp_name] = hp_distribution

        for original, translated in translations.items():
            params[translated] = params.pop(original)
            dists[translated] = dists.pop(original)

        trial.params = params
        trial.distributions = dists

        # Persist the changes to the storage (in a new study).
        newstudy.add_trial(trial)

    return newstudy
