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
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sortedness.attempts import cau, wlossf11
from torch import from_numpy, set_num_threads, tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

f = wlossf11
# cuda.is_available = lambda: False

threads = 1
set_num_threads(threads)
n = 200
k, gamma = n, 4
smooothness_ranking, smooothness_tau, decay = [5], [5], 0
batch_size = [20]
update = 1
gpu = not True
a = 20
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
W = torch.from_numpy(PCA(n_components=2).fit_transform(datax).astype(np.float32))
datay = digits.target

print(datax.shape)
print(datay.shape)

alph = datay  # "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ax = [0, 0]

torch.manual_seed(seed)
X = datax[:n]
idxs = list(range(n))
X = X.astype(np.float32)


class Dt(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return idx


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], a), torch.nn.ReLU(),
            torch.nn.Linear(a, 2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, a), torch.nn.ReLU(),
            torch.nn.Linear(a, X.shape[1])
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
D = pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).view(n, n)
w = cau(tensor(range(k)), gamma=gamma)
if gpu:
    w = w.cuda()
loss, ref = f(D, R, smooothness_ranking[0], smooothness_tau[0], ref=True, k=k, gamma=gamma, w=w)
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
    loader = [DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0], num_workers=threads, pin_memory=gpu)]
else:
    loader = [DataLoader(Dt(T, R), shuffle=True, batch_size=batch_size[0], num_workers=1, pin_memory=gpu)]
loss_fn = MSELoss()
# for idx in loader[0]:
#     encoded = model(T)[idx]
#     expected_batch = W[idx]
#     loss = loss_fn(encoded, expected_batch)
#     optimizer.zero_grad()
#     (-loss).backward()
#     optimizer.step()


def animate(i):
    c[0] += 1
    i = c[0]
    optimizer.zero_grad()
    for idx in loader[0]:
        encoded = model(T)
        expected_ranking_batch = R[idx]
        D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
        loss, ref_ = f(D_batch, expected_ranking_batch, smooothness_ranking[0], smooothness_tau[0], ref=i % update == 0, k=k, gamma=gamma, w=w)
        if ref_ != 0:
            ref = ref_
        (-loss).backward()
        optimizer.step()

    if i % update == 0:
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

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
