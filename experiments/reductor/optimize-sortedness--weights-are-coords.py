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
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import animation
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sortedness.attempts import wlossf8, cau
from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader

f = wlossf8
n = 115
k, gamma = 17, 4
smooothness_ranking, smooothness_tau, decay = 1, 1, 0
batch_size = 8
update = 10
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
seed = 0
delta = 0
rad = 80
alpha = 0.3
idxs = list(range(n))
fs = 16

digits = datasets.load_digits()
n_samples = len(digits.images)
datax = digits.images.reshape((n_samples, -1))
datax = StandardScaler().fit_transform(datax)
datay = digits.target
alph = datay
ax = [0, 0]
torch.manual_seed(seed)
X = datax[:n]
# X = X.astype(np.float32)
print(X.shape)


class Dt(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return idx

Z = PCA(n_components=2).fit_transform(X)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.from_numpy(Z), requires_grad=True)
        # torch.nn.init.uniform_(self.proj, -1, 1.)

    def forward(self):
        return self.proj


model = M()
R = from_numpy(rankdata(cdist(X, X), axis=1))

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
encoded = model()
D = pdist(encoded.unsqueeze(1), encoded.unsqueeze(0)).view(n, n)
w = cau(tensor(range(n)), gamma=gamma)


def plot(side, i, loss, ref):
    ax[side].cla()
    xcp = encoded.detach().cpu().numpy()
    ax[side].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=alph[idxs], alpha=alpha)
    for j in range(min(n, 50)):  # xcp.shape[0]):
        ax[side].text(xcp[j, 0] + delta, xcp[j, 1] + delta, alph[j], size=fs)
    ax[0].title.set_text(f"{0}:    {ref:.8f}   ")
    print(f"{i:09d}:\toptimized sur: {loss:.8f}\tresulting wtau: {ref}\t\t{smooothness_ranking:.6f}\t{smooothness_tau:.6f}")


loss, ref = f(D, R, smooothness_ranking, smooothness_tau, ref=True, k=k, gamma=gamma, w=w)
plot(0, 0, loss, ref)
optimizer = optim.RMSprop(model.parameters())
# optimizer = optim.Rprop(model.parameters())
model.train()
c = [0]
loader = DataLoader(Dt(D), shuffle=True, batch_size=batch_size)


def animate(i, smooothness_ranking, smooothness_tau):
    c[0] += 1
    i = c[0]
    for idx in loader:
        encoded = model()
        expected_ranking_batch = R[idx]
        D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
        loss, ref_ = f(D_batch, expected_ranking_batch, smooothness_ranking, smooothness_tau, ref=i % update == 0, k=k, gamma=gamma, w=w)
        if ref_ != 0:
            ref = ref_
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

    if i % update == 0:
        plot(1, i, loss, ref)
        smooothness_ranking *= 1 - decay
        smooothness_tau *= 1 - decay

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, partial(animate, smooothness_ranking=smooothness_ranking, smooothness_tau=smooothness_tau))
plt.show()
