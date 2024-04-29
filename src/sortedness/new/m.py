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


import torch

activation_function_dct = {"tanh": torch.nn.Tanh, "sigm": torch.nn.Sigmoid, "relu": torch.nn.ReLU, "relu6": torch.nn.ReLU6}
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.000000001)


class M(torch.nn.Module):
    def __init__(self, X, d, hidden_layers, activation_functions):
        """
        :param X:
                Matrix with an instance per row (often a high-dimensional data matrix).
                Only `X.shape[1]` is actually used.
        :param d:
            Target dimensionality.
        :param hidden_layers:
            Artificial neural network topology.
        :param activation_functions:
            One function for each layer defined in `hidden_layers`.
        """
        super().__init__()
        self.X = X
        self.d = d
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions
        layers = [torch.nn.Linear(X.shape[1], hidden_layers[0]), activation_function_dct[activation_functions[0]]()]
        previous = hidden_layers[0]
        for neurons, af in zip(hidden_layers[1:], activation_functions[1:]):
            layers.extend([torch.nn.Linear(previous, neurons), activation_function_dct[af]()])
            previous = neurons
        layers.append(torch.nn.Linear(previous, d))
        self.encoder = torch.nn.Sequential(*layers)
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(d, neurons), torch.nn.ReLU(),
        #     torch.nn.Linear(neurons, X.shape[1])
        # )

    def forward(self, x):
        return self.encoder(x)

    def clone(self, X=None, d=None, hidden_layers=None, activation_functions=None):
        return M(X or self.X, d or self.d, hidden_layers or self.hidden_layers, activation_functions or self.activation_functions)
