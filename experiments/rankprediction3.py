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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib import animation
from numpy import array, arange, pi, cos, sin
from numpy import random
from scipy.spatial.distance import cdist
from scipy.stats import weightedtau, rankdata
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from sortedness.embedding.tnse import wlossf2
from sortedness.local import remove_diagonal

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)

seed = 0
rnd = random.default_rng(seed)
n_epochs = 20
gpu = not True
n = 200
lines = False
letters = True
fs = 16
rad = 120
alpha = 0.5
delta = 0
alph = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ax = [0, 0]


def spiral(n):
    a = .11
    phi = arange(0, 100 * pi, 0.39)
    x = a * phi * cos(phi)
    y = a * phi * sin(phi)
    # theta = np.radians(np.linspace(0, 360 * 2, n))
    # r = theta ** 2 / 150
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    return array(list(zip(x, y)))[:n]


# x = rnd.uniform(-1, 1, n)
# y = rnd.uniform(-1, 1, n)
# X = np.vstack((x, y)).T

# with open("/home/davi/git/sortedness/mam.json") as fd:
#     X = array(json.load(fd))

X = spiral(n)

# X = fetch_covtype(return_X_y=True)[0]

torch.manual_seed(seed)
rnd.shuffle(X)
X = X[:n]
# v_min, v_max = X.min(axis=0), X.max(axis=0)
# new_min, new_max = array([-1.] * X.shape[1]), array([1.] * X.shape[1])
# X = (X - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
X = StandardScaler().fit_transform(X).astype(np.float32)


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


# I → a → 2 → a → O
a = 20


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], a), torch.nn.Sigmoid(),
            # torch.nn.Linear(a, a), torch.nn.Sigmoid(),
            torch.nn.Linear(a, 2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, a), torch.nn.Sigmoid(),
            # torch.nn.Linear(a, a), torch.nn.Sigmoid(),

            # torch.nn.Linear(a, X.shape[1])
            torch.nn.Linear(a, n - 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = M()
if gpu:
    model.cuda()
print(X.shape)
# D = from_numpy(remove_diagonal(cdist(X, X))).cuda() if gpu else torch.from_numpy(remove_diagonal(cdist(X, X)))
R = from_numpy(remove_diagonal(rankdata(cdist(X, X), axis=1))).cuda() if gpu else from_numpy(remove_diagonal(rankdata(cdist(X, X), axis=1)))
# T = tensor(X, dtype=float32).cuda()
T = from_numpy(X).cuda() if gpu else from_numpy(X)

# loader = DataLoader(Dt(T, D), shuffle=True, batch_size=15)
# loader = DataLoader(Dt(T, T), shuffle=True, batch_size=10)
loader = DataLoader(Dt(T, R), shuffle=True, batch_size=10)

optimizer = optim.RMSprop(model.parameters())
model.train()
fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
idxs = list(range(n))
ax[0].scatter(X[:, 0], X[:, 1], s=rad, c=2 * array(idxs), alpha=alpha)
if lines:
    ax[0].plot(X[:, 0], X[:, 1], alpha=alpha)
if letters:
    for i in range(min(62, X.shape[0])):
        ax[0].text(X[i, 0] + delta, X[i, 1] + delta, alph[i], size=fs)
xlim = list(ax[0].get_xlim())
ylim = list(ax[0].get_ylim())

loss_fn = MSELoss().cuda() if gpu else MSELoss()
wtau = [0]
w = tensor([1 / (1 + r) for r in range(n - 1)], requires_grad=True).cuda() if gpu else tensor([1 / (1 + r) for r in range(n - 1)], requires_grad=True)


def animate(i):
    Z, _ = model(T)
    ax[1].cla()
    xcp = Z.detach().cpu().numpy()
    ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
    if lines:
        ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
    if letters:
        for j in range(min(62, xcp.shape[0])):
            ax[1].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)

    for input_batch, expected_r_batch, _ in loader:
        _, pred_r_batch = model(input_batch)

        # loss = lossf2(pred_r_batch, expected_r_batch)
        loss = wlossf2(pred_r_batch, expected_r_batch, w)
        # loss = sum(-spearman(pr.view(1,pr.size(0)), ex.view(1,pr.size(0)), w) for pr, ex in zip(pred_r_batch, expected_r_batch))
        # loss = loss_fn(pred_r_batch, expected_r_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 3 == 0:
        wtau[0] = 0
        ds = remove_diagonal(pdist(Z.unsqueeze(1), Z.unsqueeze(0)).detach().cpu().numpy())
        for pred, target in zip(ds, R.detach().cpu().numpy()):
            wtau[0] += weightedtau(pred, target)[0]
    plt.title(f"{i}:    {wtau[0] / n:.8f}   ", fontsize=20)

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
