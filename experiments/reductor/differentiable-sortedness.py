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
from numpy import random
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, set_num_threads, tensor
from torch.utils.data import DataLoader

from sortedness.embedding.sortedness_ import Dt
from sortedness.embedding.surrogate import cau, loss_function

# relative_diff = False
bal = 0.5
gamma = 4
k, gk = 17, "sqrt"
alpha = 0.5
smooothness_tau = 1
neurons = 30
# epochs = 100
batch_size = 20
seed = 0
gpu = False

n = 1797 // 6
threads = 1
# cuda.is_available = lambda: False
set_num_threads(threads)
update = 1
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
chars = True
char_size = 16
radius = 120

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target
alphabet = datay
print(datax.shape, datay.shape)
ax = [0, 0]
torch.manual_seed(seed)
rnd = random.default_rng(seed)
X = datax[:n]
idxs = list(range(n))
X = X.astype(np.float32)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], neurons), torch.nn.Tanh(),
            torch.nn.Linear(neurons, 2)
        )
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(2, neurons), torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, X.shape[1])
        # )

    def forward(self, x):
        return self.encoder(x)


model = M()
if gpu:
    model.cuda()
print(X.shape)
Dtarget = cdist(X, X)
Dtarget = from_numpy(Dtarget / np.max(Dtarget, axis=1))
if gpu:
    Dtarget = Dtarget.cuda()
# R = from_numpy(rankdata(cdist(X, X), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(X, X), axis=1))
T = from_numpy(X).cuda() if gpu else from_numpy(X)
ca = cau(tensor(range(n)), gamma=gamma) / 0.54
print(sum(ca))
w = ca.cuda() if gpu else ca
# wharmonic = har(tensor(range(n)))

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
ax[0].cla()

xcp = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=300, n_jobs=-1).fit_transform(X)
D = from_numpy(rankdata(cdist(xcp, xcp), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(xcp, xcp), axis=1))
loss, loss_local, loss_global, ref_local, ref_global = loss_function(D, Dtarget, k, gk, w, bal, smooothness_tau, ref=True)

ax[0].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=alpha)
for j in range(min(n, 50)):  # xcp.shape[0]):
    ax[0].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)
ax[0].title.set_text(f"{0}:  {ref_local:.4f}  {ref_global:.4f}")
print(f"{0:09d}:\toptimized sur: {loss:.4f}  local/globa: {loss_local:.4f} {loss_global:.4f}  REF: {ref_local:.4f} {ref_global:.4f}\t\t{smooothness_tau:.6f}")

optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.ASGD(model.parameters())
# optimizer = optim.Rprop(model.parameters())
model.train()

c = [0]

if threads > 1:
    loader = [DataLoader(Dt(T), shuffle=True, batch_size=batch_size, num_workers=threads, pin_memory=gpu)]
else:
    loader = [DataLoader(Dt(T), shuffle=True, batch_size=batch_size, pin_memory=gpu)]


def animate(i):
    encoded = loss = loss_local = loss_global = ref_local = ref_global = None
    c[0] += 1
    i = c[0]
    for idx in loader[0]:
        encoded = model(T)
        expected_ranking_batch = Dtarget[idx]
        D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
        loss, loss_local, loss_global, ref_local, ref_global = loss_function(D_batch, expected_ranking_batch, k, gk, w, bal, smooothness_tau, ref=i % update == 0)
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

    if i % update == 0:
        ax[1].cla()
        xcp = encoded.detach().cpu().numpy()
        ax[1].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=alpha)
        if chars:
            for j in range(min(n, 50)):  # xcp.shape[0]):
                ax[1].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)
        plt.title(f"{i}:  {ref_local:.4f}  {ref_global:.4f}", fontsize=16)
    print(f"{i:09d}:\toptimized sur: {loss:.4f}  local/globa: {loss_local:.4f} {loss_global:.4f}  REF: {ref_local:.4f} {ref_global:.4f}\t\t{smooothness_tau:.6f}")

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
