import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib import animation
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import from_numpy, tensor
from torch.utils.data import DataLoader

from sortedness.embedding.sortedness_ import Dt
from sortedness.embedding.surrogate import cau, loss_function

alpha = 0.5
beta = 0.5
gamma = 4
k, gk = 17, "sqrt"
lambd = 0.1
batch_size = 20
seed = 0
gpu = False

n = 1797 // 6
threads = 1
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
X = datax[:n]
idxs = list(range(n))
X = X.astype(np.float32)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.empty(*X.shape), requires_grad=True)
        self.proj = torch.nn.init.uniform_(self.proj, -1, 1.)

    def forward(self):
        return self.proj


model = M()
if gpu:
    model.cuda()
print(X.shape)
Dtarget = cdist(X, X)
Dtarget = from_numpy(Dtarget / np.max(Dtarget))
# Dtarget = from_numpy(Dtarget)
if gpu:
    Dtarget = Dtarget.cuda()
# R = from_numpy(rankdata(cdist(X, X), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(X, X), axis=1))
T = from_numpy(X).cuda() if gpu else from_numpy(X)
w = cau(tensor(range(n)), gamma=gamma).cuda() if gpu else cau(tensor(range(n)), gamma=gamma)
# wharmonic = har(tensor(range(n)))

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
ax[0], ax[1] = axs
ax[0].cla()

xcp = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=300, n_jobs=-1).fit_transform(X)
D = from_numpy(rankdata(cdist(xcp, xcp), axis=1)).cuda() if gpu else from_numpy(rankdata(cdist(xcp, xcp), axis=1))
loss, loss_local, loss_global, ref_local, ref_global = loss_function(Dtarget, D, None, None, k, gk, w, alpha, beta, lambd, ref=True)

ax[0].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=0.5)
for j in range(min(n, 50)):  # xcp.shape[0]):
    ax[0].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)
ax[0].title.set_text(f"{0}:  {ref_local:.4f}  {ref_global:.4f}")
print(f"{0:09d}:\toptimized sur: {loss:.4f}  local/globa: {loss_local:.4f} {loss_global:.4f}  REF: {ref_local:.4f} {ref_global:.4f}\t\t{lambd:.6f}")

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
        encoded = model()
        expected_ranking_batch = Dtarget[idx]
        D_batch = pdist(encoded[idx].unsqueeze(1), encoded.unsqueeze(0)).view(len(idx), -1)
        loss, loss_local, loss_global, ref_local, ref_global = loss_function(Dtarget, D, None, None, k, gk, w, alpha, beta, lambd, ref=i % update == 0)
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

    if i % update == 0:
        ax[1].cla()
        xcp = encoded.detach().cpu().numpy()
        ax[1].scatter(xcp[:, 0], xcp[:, 1], s=radius, c=alphabet[idxs], alpha=0.5)
        if chars:
            for j in range(min(n, 50)):  # xcp.shape[0]):
                ax[1].text(xcp[j, 0], xcp[j, 1], alphabet[j], size=char_size)
        plt.title(f"{i}:  {ref_local:.4f}  {ref_global:.4f}", fontsize=16)
    print(f"{i:09d}:\toptimized sur: {loss:.4f}  local/globa: {loss_local:.4f} {loss_global:.4f}  REF: {ref_local:.4f} {ref_global:.4f}\t\t{lambd:.6f}")

    return ax[1].step([], [])


mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
with torch.enable_grad():
    anim = animation.FuncAnimation(fig, animate)
plt.show()
