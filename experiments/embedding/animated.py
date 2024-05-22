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
from matplotlib import animation
from numpy import random
from scipy.stats import halfnorm
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, set_num_threads, tensor
from torch.optim import RMSprop
from torch.utils.data import DataLoader

from sortedness.embedding.sigmas_ import findweight, findsigma
from sortedness.embedding.sortedness_ import Dt, step

# X     000000018:	optimized sur: 0.1706  local/globa: 0.1550 0.1879  REF: 0.6209 0.5486		1.000000
# both  000000018:	optimized sur: 0.1663  local/globa: 0.1611 0.1735  REF: 0.6426 0.4954		1.000000
# X_    000000018:	optimized sur: 0.1641  local/globa: 0.1556 0.1751  REF: 0.5480 0.4830		1.000000
n = 410  # 1797
ref = True
alpha = 0
beta = 0
gamma = 4
sigma = 20
pct, kappa = 90, 5
K, gk = 20, 400
lambd = .05
neurons = 20
batch_size = 20
seed = 0
gpu = False

# wf = cau(tensor(range(n)), gamma=gamma) / 0.54
# wf = tensor(gaussian_np(list(range(n)), sigma=sigma))
# wharmonic = har(tensor(range(n)))
sigma_ = findsigma(pct, kappa)
k = int(halfnorm.ppf(.9999, 0, sigma_))
wf = tensor([findweight(x, sigma_) for x in range(k)])
print(k, float(sum(wf)))

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

X = X[:, ~(X == X[0, :]).all(0)]
mn = X.min(axis=0)
mx = X.max(axis=0)
X = (X - mn) / (mx - mn)

idxs = list(range(n))
X = X.astype(np.float32)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], neurons), torch.nn.Tanh(),
            # torch.nn.Linear(neurons, neurons), torch.nn.ReLU(),
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
w = wf.cuda() if gpu else wf

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
# ax[0].cla()

print("tsne")
xcp = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=300, n_jobs=-1).fit_transform(X)
print("done")
X_ = xcp
X = from_numpy(X).cuda() if gpu else from_numpy(X)

# Dsorted, idxs_by_D = (None, None) if alpha == 1 else topk(D, k, largest=False, dim=1)

# ax[0].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=0.5)
# for j in range(min(n, 50)):  # xcp.shape[0]):
#     ax[0].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)

optimizer = RMSprop(model.parameters())
model.train()

c = [0]

if threads > 1:
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, num_workers=threads, pin_memory=gpu)
else:
    loader = DataLoader(Dt(X), shuffle=True, batch_size=batch_size, pin_memory=gpu)


def animate(i):
    print(i)
    for idx in loader:
        X_, quality = step(X, idx, model, gpu, k, K, w, alpha, beta, lambd, optimizer)
    ax[0].cla()
    xcp = X_.detach().cpu().numpy()
    ax[0].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=0.5)
    if chars:
        for j in range(min(n, 50)):  # xcp.shape[0]):
            ax[0].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)
    return ax[0].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
