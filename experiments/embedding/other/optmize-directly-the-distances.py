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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib import animation
from numpy import random
from scipy.spatial.distance import cdist
from scipy.stats import cauchy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sortedness.attempts import lossf
from torch.utils.data import Dataset

from sortedness.local import remove_diagonal

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target


ae = False
gpu = False
batch_size = 20
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
seed = 1
n = 300
lines = False
letters = True
fs = 16
rad = 120
alpha = 0.5
delta = 0
alph = datay  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ax = [0, 0]
cau = cauchy(0).pdf

torch.manual_seed(seed)
rnd = random.default_rng(seed)
X = datax
rnd = random.default_rng(seed)
X = X[:n]
idxs = list(range(n))
X = X.astype(np.float32)
print(X.shape)
print(datay.shape)
R = X


class Dt(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target, idx


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.Tensor(n, 2), requires_grad=True)
        torch.nn.init.uniform_(self.proj, -1, 1.)

    def forward(self, x=None):
        x = self.proj
        xc = x
        ax[1].cla()
        xcp = xc.cpu().detach().numpy()
        ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
        if lines:
            ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
        if letters:
            for i in range(min(62, xc.shape[0])):
                ax[1].text(xcp[i, 0] + delta, xcp[i, 1] + delta, alph[i], size=fs)

        ds = pdist(x.unsqueeze(1), x.unsqueeze(0))
        ds = ds.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        return ds


m = M()
if gpu:
    m.cuda()
m.train()

optim = torch.optim.RMSprop(m.parameters())

D = torch.from_numpy(remove_diagonal(cdist(R, R))).cuda() if gpu else torch.from_numpy(remove_diagonal(cdist(R, R)))


def animate(i):
    Z = m(R)
    ax[1].cla()

    optim.zero_grad()
    lo = lossf(Z, D, smoothness=10, i=i)
    lo.backward()
    optim.step()

    xcp = Z.detach().cpu().numpy()

    ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=alph[idxs], alpha=alpha)
    # ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
    if lines:
        ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
    if letters:
        for j in range(30):  # xcp.shape[0]):
            ax[1].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)

    return ax[1].step([], [])


fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
