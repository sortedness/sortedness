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

import json

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import animation
from numpy import array
from numpy import random
from scipy.spatial.distance import cdist
from scipy.stats import weightedtau
from torch import from_numpy, float32
from torch import tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from sortedness.local import remove_diagonal

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)

seed = 0
n_epochs = 2000
gpu = not True
n = 300
lines = not True
letters = True
fs = 16
rad = 120
alpha = 0.5
delta = 0
alph = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ax = [0, 0]
# Read data, convert to NumPy arrays
with open("/home/davi/git/sortedness/mam.json") as fd:
    R = array(json.load(fd))
rnd = random.default_rng(seed)
rnd.shuffle(R)
R = R[:n]
v_min, v_max = R.min(axis=0), R.max(axis=0)
new_min, new_max = array([-1.] * 3), array([1.] * 3)
R = (R - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
gpu = not True


class Dt(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target

l=20
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, l), torch.nn.Sigmoid(),
            # torch.nn.Linear(16, 16), torch.nn.Sigmoid(),
            torch.nn.Linear(l, 2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, l), torch.nn.Sigmoid(),
            # torch.nn.Linear(16, 16), torch.nn.Sigmoid(),
            torch.nn.Linear(l, n - 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


model = M()
if gpu:
    model.cuda()
D = from_numpy(remove_diagonal(cdist(R, R))).cuda() if gpu else torch.from_numpy(remove_diagonal(cdist(R, R)))
print(R.shape, D.shape)
dataset = Dt(R, D)
trainset = dataset
# trainset, testset = random_split(dataset, [0.7, 0.3])
loader = DataLoader(trainset, shuffle=True, batch_size=5)

optimizer = optim.RMSprop(model.parameters())
model.train()
fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
idxs = list(range(n))
ax[0].scatter(R[:, 0], R[:, 1], s=rad, c=2 * array(idxs), alpha=alpha)
if lines:
    ax[0].plot(R[:, 0], R[:, 1], alpha=alpha)
if letters:
    for i in range(min(62, R.shape[0])):
        ax[0].text(R[i, 0] + delta, R[i, 1] + delta, alph[i], size=fs)
xlim = list(ax[0].get_xlim())
ylim = list(ax[0].get_ylim())

loss_fn = MSELoss()


# for epoch in range(n_epochs):
def animate(i):
    for X_batch, y_batch in loader:
        enc, dec = model(X_batch)
        loss = loss_fn(dec, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # encs = vstack([model(X_batch)[0] for X_batch, _ in loader])
    # ds = pdist(encs.unsqueeze(1), encs.unsqueeze(0))
    # ds = ds.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
    # loss = lossf(ds, D)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    enc, _ = model(tensor(R, dtype=float32))
    ax[1].cla()
    xcp = enc.detach().numpy()
    ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
    if lines:
        ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
    if letters:
        for i in range(min(62, xcp.shape[0])):
            ax[1].text(xcp[i, 0] + delta, xcp[i, 1] + delta, alph[i], size=fs)

    wtau = 0
    enc, _ = model(tensor(R, dtype=float32))
    ds = remove_diagonal(pdist(enc.unsqueeze(1), enc.unsqueeze(0)).detach().cpu().numpy())
    for pred, target in zip(ds, D.detach().cpu().numpy()):
        wtau += weightedtau(pred, target)[0]
    plt.title(f"{i}:    {wtau / n:.8f}   ", fontsize=20)

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
