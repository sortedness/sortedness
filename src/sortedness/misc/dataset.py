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

import glob
import os
from pathlib import Path

import numpy as np


def load_dataset(dataset_name, result=False):
    dir = f"{Path.home()}/csv_proj_sortedness_out"
    data_dir = os.path.join(dir, dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    if result:
        res = f"{Path.home()}/csvs/*X_*{dataset_name}*.csv"
        list_of_files = glob.glob(res)
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file, "loaded!!!")
        X_ = np.fromfile(latest_file, sep=',')
        return X, y, X_.reshape((X.shape[0], 2))

    return X, y
