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
from scipy.stats import weightedtau, rankdata
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sortedness.attempts import wlossf4, cau, wlossf5
from torch import from_numpy, tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from sortedness.local import remove_diagonal

n = 30
smoothness_ranking, lambd = 0.1, 0.1
# smoothness_ranking = lambd = 0.0000000001
update = 10
ae = False
gpu = not True
a = 20
batch_size = 20
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
seed = 0
lines = False
letters = True
fs = 16
rad = 120
alpha = 0.5
delta = 0

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target


print(datax.shape)
print(datay.shape)

alph = datay  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ax = [0, 0]


torch.manual_seed(seed)
rnd = random.default_rng(seed)
X = datax
rnd = random.default_rng(seed)
X = X[:n]
idxs = list(range(n))
X = X.astype(np.float32)


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


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], a), torch.nn.ReLU(),
            torch.nn.Linear(a, 2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, a), torch.nn.ReLU(),
            torch.nn.Linear(a, X.shape[1]) if ae else torch.nn.Linear(a, n - 1)
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
loader = DataLoader(Dt(T, T), shuffle=True, batch_size=batch_size) if ae else DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size)

optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.Rprop(model.parameters())
model.train()
fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
# ax[0].scatter(X[:, 0], X[:, 1], s=rad, c=2 * array(idxs), alpha=alpha)
ax[0].scatter(X[:, 0], X[:, 1], s=rad, c=alph[idxs], alpha=alpha)
if lines:
    ax[0].plot(X[:, 0], X[:, 1], alpha=alpha)
if letters:
    for i in range(30):  # X.shape[0]):
        ax[0].text(X[i, 0] + delta, X[i, 1] + delta, alph[i], size=fs)
xlim = list(ax[0].get_xlim())
ylim = list(ax[0].get_ylim())

loss_fn = MSELoss().cuda() if gpu else MSELoss()
wtau, resur = [0], [0]


# w = tensor([1 / (1 + r) for r in range(n - 1)], requires_grad=True).cuda() if gpu else tensor([1 / (1 + r) for r in range(n - 1)], requires_grad=True)


# xcp = T.detach().cpu().numpy()
# tsne= TSNE(random_state = 42, n_components=2, verbose=0, perplexity=40, n_iter=300)
# xcp = tsne.fit_transform(xcp)

def animate(i):
    c = lo = 0
    for input_batch, expected_r_batch, _ in loader:
        _, pred_r_batch = model(input_batch)

        if ae:
            loss = loss_fn(pred_r_batch, expected_r_batch)
        else:
            # loss = lossf2(pred_r_batch, expected_r_batch)
            # loss = wlossf4(pred_r_batch, expected_r_batch, smoothness_ranking, lambd)
            loss = wlossf5(pred_r_batch, expected_r_batch, lambd)
        lo += float(loss)
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()
        c+=1
    lo /= c

    # encoded, predicted_rankings = model(T)
    # # lo = wlossf4(predicted_rankings, R, smoothness_ranking, lambd)
    # lo = wlossf5(predicted_rankings, R, lambd)
    # optimizer.zero_grad()
    # (-lo).backward()
    # optimizer.step()

    if i % update == 0:
        encoded, predicted_rankings = model(T)
        ax[1].cla()

        wtau[0], resur[0] = 0, 0
        ds = remove_diagonal(pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).detach().cpu().numpy())
        c = 0
        for pred, target in zip(ds, R.detach().cpu().numpy()):
            wtau[0] += weightedtau(-target, -pred, weigher=cau)[0]
            resur[0] += wlossf4(tensor([target]), tensor([pred]), smoothness_ranking, lambd)
            c += 1
        wtau[0] = wtau[0] / c
        resur[0] = resur[0] / c

        xcp = encoded.detach().cpu().numpy()

        ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=alph[idxs], alpha=alpha)
        # ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
        if lines:
            ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
        if letters:
            for j in range(30):  # xcp.shape[0]):
                ax[1].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)
        plt.title(f"{i}:    {wtau[0]:.8f}   ", fontsize=20)
        print(f"{i:09d}:\toptimed sur: {lo:.8f}\tresulting wtau: {wtau[0]}\tresulting sur: {resur[0]}")
    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()

# não deu certo: usar cauchy e ver se surrogate_wtau vence surrogate_tau e RMSE
# depois: implementar topologia crescente
# depois: comparar com tsne, mds, pca, autoencoder; usando sortedness-cauchy e stress
# depois: comparar com tsne, mds, pca, autoencoder; usando tempo
