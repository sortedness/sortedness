#  Copyright (c) 2024. Davi Pereira dos Santos
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

import numpy as np
from hdict import hdict
from hdict.dataset.dataset import df2Xy
from numpy import array
from pandas import DataFrame
from torch import cat, manual_seed, from_numpy, sum
from torch.nn import Module, Linear, Sequential, Tanh, Sigmoid
from torch.optim import Rprop


class M(Module):
    def __init__(self, n_inputs):
        super(M, self).__init__()
        self.nested_nets = [
            Sequential(Linear(n_inputs, 1), Tanh()),
            Sequential(Linear(n_inputs, 2), Tanh(), Linear(2, 1), Tanh()),
            Sequential(Linear(n_inputs, 3), Tanh(), Linear(3, 1), Tanh()),
            Sequential(Linear(n_inputs, 2), Tanh(), Linear(2, 2), Tanh(), Linear(2, 1), Tanh()),
            # Sequential(Linear(n_inputs, 3), Tanh(), Linear(3, 2), Tanh(), Linear(2, 1), Tanh()),
            # Sequential(Linear(n_inputs, 3), Tanh(), Linear(3, 3), Tanh(), Linear(3, 1), Tanh())
        ]
        self.latent_layer = Linear(len(self.nested_nets), 1)
        self.last_layer = Sequential(self.latent_layer, Sigmoid())

    def forward(self, X):
        latent_values = cat([net(X) for net in self.nested_nets], dim=1)
        return self.last_layer(latent_values)

    def latent_weights(self):
        return self.latent_layer.weight.data


d = hdict.fromfile("/home/davi/research/abalone-3class.arff")
# X0, y0 = mnist(1000)
d.apply(df2Xy, target="c", out=("X", "y"))
X0: DataFrame
X0, y0 = d.X, d.y
X0.loc[X0["V1"] == "M", "V1"] = 1
X0.loc[X0["V1"] == "I", "V1"] = 0
X0.loc[X0["V1"] == "F", "V1"] = -1
X0, y0 = X0.to_numpy().astype(np.float32), array([float(v) for v in y0.to_numpy()]).astype(np.float32)
rnd = np.random.default_rng(10)
rnd.shuffle(X0)
rnd.shuffle(y0)
X0, y0 = from_numpy(X0), from_numpy(y0)
y0[y0 <= 2] = 0
y0[y0 > 2] = 1
w = 3000
manual_seed(0)
for h in range(0, 3):
    X0 = X0[:, rnd.permutation(X0.shape[1])]
    X0 = X0[rnd.permutation(X0.shape[0]), :]
    for i in range(0, len(y0), w):
        X, y = X0[i:i + w], y0[i:i + w]
        model = M(X.shape[1])
        model.train()
        optim = Rprop(model.parameters())
        for j in range(100):
            S = X[rnd.integers(X.shape[0], size=100)]
            loss = sum((y - model(S)) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
        z = (model(X) >= 0.5).reshape(y.shape).int()
        # noinspection PyUnresolvedReferences
        acc = (y == z).int().sum().item() / y.size(0)
        print(f"{i:4}", [f"{x:+3.2f}" for x in model.latent_weights().tolist()[0]], acc)
        # print(f"{i:4}", sorted([f"{abs(x):+3.2f}" for x in model.latent_weights().tolist()[0]]), acc)
        # print()
    print()
