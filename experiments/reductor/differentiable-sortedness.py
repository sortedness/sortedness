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
from scipy.stats import rankdata
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, set_num_threads
from torch.utils.data import Dataset, DataLoader

from sortedness.embedding.tnse import wlossf6

# cuda.is_available = lambda: False

threads = 1
set_num_threads(threads)
batches = True
n = 1150
smooothness_ranking, smooothness_tau, decay = [5], [5], 0
# delta_barch_size = 1
# min_batch_size = 1
# max_batch_size = 300
# batch_size = [min_batch_size if delta_barch_size > 0 else max_batch_size]
batch_size = [20]
update = 2
gpu = False
a = 15
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

# h = hdict.fromfile("/home/davi/train.csv")
# h.show()
# datax = h.df.loc[:, 'pixel0':].values
# datay = h.df.loc[:, 'label'].values

# mask = np.isin(datay, [1,0])
# datax = datax[mask]
# datay = datay[mask]

print(datax.shape)
print(datay.shape)

alph = datay  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
torch.manual_seed(seed)
rnd = random.default_rng(seed)
X = datax
# rnd.shuffle(X)
rnd = random.default_rng(seed)
# rnd.shuffle(y)
X = X[:n]
# y = y[:n]
idxs = list(range(n))
# idxs = y

# print("1111111111\n", np.unique(y))

# v_min, v_max = X.min(axis=0), X.max(axis=0)
# new_min, new_max = array([-1.] * X.shape[1]), array([1.] * X.shape[1])
# X = (X - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

# X = StandardScaler().fit_transform(X).astype(np.float32)
X = X.astype(np.float32)


class Dt(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        # features = self.X[idx]
        # target = self.y[idx]
        # return features, target, idx
        return idx


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], a), torch.nn.ReLU(),
            torch.nn.Linear(a, 2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, a), torch.nn.ReLU(),
            torch.nn.Linear(a, X.shape[1]),
        )

    def forward(self, x):
        return self.encoder(x)


model = M()
if gpu:
    model.cuda()
print(X.shape)
R = from_numpy(rankdata(cdist(X, X), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(X, X), axis=1))
T = from_numpy(X).cuda() if gpu else from_numpy(X)

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
encoded = model(T)
if batches:
    D = pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).view(n, n)
    loss, ref = wlossf6(D, R, smooothness_ranking[0], smooothness_tau[0], ref=True)
ax[0].cla()
xcp = encoded.detach().cpu().numpy()
ax[0].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=alph[idxs], alpha=alpha)
if lines:
    ax[0].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
if letters:
    for j in range(min(n, 50)):  # xcp.shape[0]):
        ax[0].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)
ax[0].title.set_text(f"{0}:    {ref:.8f}   ")
print(f"{0:09d}:\toptimized sur: {loss:.8f}\tresulting wtau: {ref}\t\t{smooothness_ranking[0]:.6f}\t{smooothness_tau[0]:.6f}")

optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.Rprop(model.parameters())
model.train()

c = [0]

if threads > 1:
    loader = [DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0], num_workers=1)]
else:
    loader = [DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0], num_workers=1)]


def animate(i):
    c[0] += 1
    i = c[0]
    if batches:
        for idx in loader[0]:
            encoded = model(T)
            expected_ranking_batch = R[idx]
            D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
            loss, ref = wlossf6(D_batch, expected_ranking_batch, smooothness_ranking[0], smooothness_tau[0], ref=True or i % update == 0)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
    else:
        encoded = model(T)
        D = pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).view(n, n)
        loss, ref = wlossf6(D, R, smooothness_ranking[0], smooothness_tau[0], ref=True or i % update == 0)
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

    if True or i % update == 1:
        if False and batches:
            # if threads > 1:
            #     loader[0] = DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0], num_workers=threads)
            # else:
            #     loader[0] = DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0])
            # batch_size[0] += delta_barch_size
            # if batch_size[0] < min_batch_size:
            #     batch_size[0] = min_batch_size
            # if batch_size[0] > max_batch_size:
            #     batch_size[0] = max_batch_size
            D = pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).view(n, n)
            loss, ref = wlossf6(D, R, smooothness_ranking[0], smooothness_tau[0], ref=True)
        ax[1].cla()
        xcp = encoded.detach().cpu().numpy()
        ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=alph[idxs], alpha=alpha)
        if lines:
            ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
        if letters:
            for j in range(min(n, 50)):  # xcp.shape[0]):
                ax[1].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)
        plt.title(f"{i}:    {ref:.8f}   ", fontsize=20)
        smooothness_ranking[0] *= 1 - decay
        smooothness_tau[0] *= 1 - decay
        print(f"{i:09d}:\toptimized sur: {loss:.8f}\tresulting wtau: {ref}\t\t{smooothness_ranking[0]:.6f}\t{smooothness_tau[0]:.6f}")
    elif not batches:
        print(f"{i:09d}:\toptimized sur: {loss:.8f}\tresulting wtau: {ref}\t\t{smooothness_ranking[0]:.6f}\t{smooothness_tau[0]:.6f}")

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
