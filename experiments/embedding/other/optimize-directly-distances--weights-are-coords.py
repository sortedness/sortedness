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
from time import sleep

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import animation
from numpy import array
from scipy.spatial.distance import cdist
from sortedness.attempts import lossf
from torch import tensor

from sortedness.local import remove_diagonal

gpu = not True
n = 40
lines = not True
letters = True
fs = 16
rad = 80
alpha = 0.3
delta = 0
seed = 0
np.random.seed(seed)
ax = [0, 0]
fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs

sc = [0, 0]

torch.manual_seed(seed)
rnd = numpy.random.default_rng(seed)


def spiral(n):
    a = .11
    phi = np.arange(0, 10 * np.pi, 0.39)
    x = a * phi * np.cos(phi)
    y = a * phi * np.sin(phi)
    # theta = np.radians(np.linspace(0, 360 * 2, n))
    # r = theta ** 2 / 150
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    return array(list(zip(x, y)))[:n]


R = spiral(n)

# with open("/home/davi/git/sortedness/mam.json") as fd:
#     R = array(json.load(fd))
# rnd.shuffle(R)
# R = R[:n]
# v_min, v_max = R.min(axis=0), R.max(axis=0)
# new_min, new_max = array([-1.] * 3), array([1.] * 3)
# R = (R - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

# Y = torch.from_numpy(rankdata(D, axis=1))
pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
w = tensor([1 / (1 + r) for r in range(n - 1)])
idxs = list(range(n))

alph = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
c = [0]
o = [0]
running = [True]

sc[0] = ax[0].scatter(R[:, 0], R[:, 1], s=rad, c=2 * array(idxs), alpha=alpha)
if lines:
    sc[0] = ax[0].plot(R[:, 0], R[:, 1], alpha=alpha)
if letters:
    for i in range(min(62, R.shape[0])):
        ax[0].text(R[i, 0] + delta, R[i, 1] + delta, alph[i], size=fs)
xlim = list(ax[0].get_xlim())
ylim = list(ax[0].get_ylim())


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
        sc[1] = ax[1].scatter(xcp[:, 0], xcp[:, 1], s=rad, c=idxs, alpha=alpha)
        if lines:
            sc[1] = ax[1].plot(xcp[:, 0], xcp[:, 1], alpha=alpha)
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

# optim = torch.optim.SGD(m.parameters(), lr=0.5, momentum=0.5)
# optim = torch.optim.Adagrad(m.parameters())
optim = torch.optim.Rprop(m.parameters())
# optim = torch.optim.RMSprop(m.parameters())

D = torch.from_numpy(remove_diagonal(cdist(R, R))).cuda() if gpu else torch.from_numpy(remove_diagonal(cdist(R, R)))


def animate(i):
    if not running[0]:
        if running[0] is None:
            lossf(m(), D, smoothness=10)
            print("stopped")
        running[0] = False
        sleep(1)
        return ax[1].step([], [])

    # print(i, end="\t")
    optim.zero_grad()
    lo = lossf(m(), D, 10, i, running)
    lo.backward()
    print(c[0], float(lo))
    if abs(float(lo) - o[0]) < 0.000_000_1:
        running[0] = None
    o[0] = float(lo)
    optim.step()
    # print(optim.)
    # sleep(1)
    c[0] += 1
    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()

"""
diferenciabilidade resolve empates, basta por um ruido desprezivel

para artigo latex
\begin{align*}
\tilde\lambda_{{\rho}_{\scriptscriptstyle 1,t}} \simeq
\lambda_{{\rho}_{\scriptscriptstyle 1,t}}
\end{align*}

local           global          value+ (rnd=0.0)    value* (rnd=0.5)
--------------------------------------------------------------------
perfect         perfect         1.00                1.00
perfect         half-ordered    0.75                0.73
half-ordered    perfect         0.75                0.73
half-ordered    half-ordered    0.50                0.50
perfect         random          0.50                0.41
random          perfect         0.50                0.41
half-ordered    random          0.25                0.22
random          half-ordered    0.25                0.22
random          random          0.00                0.00
random          reversed       -0.50               -1.00
reversed        random         -0.50               -1.00 
reversed        reversed       -1.00               -1.00

"""
