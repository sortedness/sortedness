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
from math import pi, sqrt

import matplotlib.pyplot as plt
import torch
from scipy.stats import weightedtau, kendalltau
from torch import tanh, sum, tensor, topk, sort
from torchsort import soft_rank

cau = lambda r, gamma=1: 1 / pi * gamma / (gamma ** 2 + r ** 2)
har = lambda r: 1 / (r + 1)


# cau = lambda r, gamma=1: 1 / (1 + r)


def pdiffs(x):
    dis = x.unsqueeze(1) - x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def psums(x):
    dis = x.unsqueeze(1) + x
    indices = torch.triu_indices(*dis.shape, offset=1)
    return dis[indices[0], indices[1]]


def surrogate_tau(a, b, smooothness):
    da, db = pdiffs(a), pdiffs(b)
    return sum(tanh(da / smooothness) * tanh(db / smooothness))  # todo: somar random residual para nunca ser zero? p/ otimizar tanto faz?


def surrogate_wtau2(a, b, wa, wb, smooothness):
    da, db, swa, swb = pdiffs(a), pdiffs(b), psums(wa), psums(wb)
    s = tanh(da / smooothness) * tanh(db / smooothness)
    return (sum(s * swa) / sum(swa) + sum(s * swb) / sum(swb)) / 2


def surrogate_wtau(a, b, w, smooothness):
    da, db, sw = pdiffs(a), pdiffs(b), psums(w)
    s = tanh(da / smooothness) * tanh(db / smooothness)
    return sum(s * sw) / sum(sw)


def lossf(predicted_D, expected_D, smooothness, i=None, running=None):
    n = predicted_D.shape[0]
    m = predicted_D.shape[1]
    r = []
    mu = wtau = 0
    for pred, target in zip(predicted_D, expected_D):  # tODO: same size? ver diagonal
        surr = surrogate_tau(pred.view(m), target.view(m), smooothness=smooothness)
        wtau += weightedtau(pred.detach().cpu().numpy(), target.detach().cpu().numpy())[0]
        mu += surr
    plt.title(f"{i}:    {wtau / predicted_D.shape[0]:.8f}    {float(mu):.8f}   {running=}", fontsize=20)
    return mu / n


def lossf2(predicted_ranks, expected_ranks, i=None, running=None):
    n, o = predicted_ranks.shape
    mu = 0
    for pred, target in zip(predicted_ranks, expected_ranks):
        mu += surrogate_tau(pred, target)

    return mu / ((o * (o - 1) / 2) * n)


def wlossf4(predicted_ranks, expected_ranks, smooothness_ranking, smooothness_tau):
    n, o = predicted_ranks.shape
    mu = 0
    for pred, target in zip(predicted_ranks, expected_ranks):
        w_p = tensor([cau(r, gamma=4) for r in soft_rank(pred.view(1, o), regularization_strength=smooothness_ranking).view(o)], requires_grad=True)
        w_t = tensor([cau(r, gamma=4) for r in soft_rank(target.view(1, o), regularization_strength=smooothness_ranking).view(o)], requires_grad=True)
        mu += surrogate_wtau2(pred, target, w_p, w_t, smooothness_tau)
    return mu / n


def wlossf5(predicted_ranks, expected_ranks, smooothness_tau):
    n, o = predicted_ranks.shape
    mu = 0
    for pred, target in zip(predicted_ranks, expected_ranks):
        w_p = tensor([cau(r, gamma=4) for r in pred], requires_grad=True)  # .cuda()
        w_t = tensor([cau(r, gamma=4) for r in target], requires_grad=True)  # .cuda()
        mu += surrogate_wtau2(pred, target, w_p, w_t, smooothness_tau)
    return mu / n


def wlossf6(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w):
    n, o = predicted_D.shape
    mu = tau = tau_ = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        w_p = cau(soft_rank(pred_d.view(1, o), regularization_strength=smooothness_ranking).view(o), gamma=gamma)
        w_t = cau(target_r, gamma=4)  # todo: pesos do target ja podem vir prontos de fora (e serem ordenados como estou fdazendo aqui com os ranks)
        # mu += surrogate_wtau(pred_d, target_r, w_p, smooothness_tau)
        mu += surrogate_wtau2(pred_d, target_r, w_p, w_t, smooothness_tau)
        if ref:
            # a, idxs = topk(pred_d, k, largest=False)
            # b = target_r[idxs]
            # a_, idxs = topk(target_r, k, largest=False)
            # b_ = pred_d[idxs]

            # wtau0 = weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=False)[0]
            # wtau0 = weightedtau(pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=None)[0]
            wtau0 = weightedtau(-pred_d.cpu().detach().numpy(), -target_r.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma))[0]

            # wtau1 = weightedtau(a_.cpu().detach().numpy(), b_.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=False)[0]
            # tau_local = (wtau0 + wtau1) / 2
            # tau_global = kendalltau(pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy())[0]
            tau += wtau0  # sqrt((tau_local + 1) / 2 * (tau_global + 1) / 2) * 2 - 1
    return mu / n, tau / n


def wlossf8(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w):
    n, o = predicted_D.shape
    mu = tau_local = tau_global = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        # pmn, tmn = torch.min(pred_d), torch.min(target_r)
        # pmx, tmx = torch.max(pred_d), torch.max(target_r)
        # pred_d = (pred_d - pmn) / (pmx - pmn)
        # target_r = (target_r - tmn) / (tmx - tmn)

        # w_p = cau(soft_rank(pred_d.view(1, o), regularization_strength=smooothness_ranking).view(o), gamma=gamma)
        # w_t = cau(target_r, gamma=4)  # todo: pesos do target ja podem vir prontos de fora (e serem ordenados como estou fdazendo aqui com os ranks)
        # mu += surrogate_wtau2(pred_d, target_r, w_p, w_t, smooothness_tau)

        a, idxs = sort(pred_d)
        b = target_r[idxs]

        # a, idxs = topk(pred_d, k, largest=False)
        # b = target_r[idxs]
        # a_, idxs = topk(target_r, k, largest=False)
        # b_ = pred_d[idxs]

        surrtau = surrogate_wtau(a, b, w, smooothness_tau)  # todo: empates, diagonal, normalização p/ compatibilizar X com X_
        # surrtau_ = surrogate_wtau(a_, b_, w, smooothness_tau)
        # mu_local = (surrtau + surrtau_) / 2
        # mu_global = surrogate_tau(pred_d, target_r, smooothness_tau) / (o ** 2 - o) * 2
        mu += surrtau  # torch.sqrt((mu_local + 1) / 2 * (mu_global + 1) / 2) * 2 - 1

        if ref:
            tau_local += weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=False)[0]
            # tau+=tau_local
            tau_global += kendalltau(pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy())[0]
            # tau += sqrt((tau_local + 1) / 2 * (tau_global + 1) / 2) * 2 - 1

    return mu / n, tau_local / n, tau_global / n


def wlossf10(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w):
    n, o = predicted_D.shape
    mu = tau = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        w = cau(soft_rank(pred_d.view(1, o), regularization_strength=smooothness_ranking).view(o), gamma=gamma)
        mu_local = surrogate_wtau(pred_d, target_r, w, smooothness_tau)
        mu_global = surrogate_tau(pred_d, target_r, smooothness_tau) / (o ** 2 - o) * 2
        mu += torch.sqrt((mu_local + 1) / 2 * (mu_global + 1) / 2) * 2 - 1

        if ref:
            a, b = pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy()
            tau_local = weightedtau(a, b, weigher=partial(cau, gamma=gamma), rank=False)[0]
            tau_global = kendalltau(a, b)[0]
            tau += sqrt((tau_local + 1) / 2 * (tau_global + 1) / 2) * 2 - 1

    return mu / n, tau / n


def wlossf11(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w):
    n, o = predicted_D.shape
    mu = tau = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        w = cau(soft_rank(pred_d.view(1, o), regularization_strength=smooothness_ranking).view(o), gamma=gamma)
        mu_local = surrogate_wtau(pred_d, target_r, w, smooothness_tau)
        mu += mu_local

        if ref:
            a, b = pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy()
            tau_local = weightedtau(a, b, weigher=partial(cau, gamma=gamma), rank=False)[0]
            tau += tau_local

    return mu / n, tau / n


def wlossf12(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w):
    n, o = predicted_D.shape
    mu = tau = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        a, idxs = topk(pred_d, k, largest=False)
        b = target_r[idxs]
        mu_local = surrogate_wtau(a, b, w[:k], smooothness_tau)
        mu += mu_local

        if ref:
            tau_local = weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=False)[0]
            tau += tau_local

    return mu / n, tau / n


def wlossf9(predicted_D, expected_R, smooothness_ranking, smooothness_tau, ref, k, gamma, w, wharmonic):
    n, o = predicted_D.shape
    mu = tau_local = tau_global = 0
    for pred_d, target_r in zip(predicted_D, expected_R):
        # pmn, tmn = torch.min(pred_d), torch.min(target_r)
        # pmx, tmx = torch.max(pred_d), torch.max(target_r)
        # pred_d = (pred_d - pmn) / (pmx - pmn)
        # target_r = (target_r - tmn) / (tmx - tmn)

        # a, idxs = topk(pred_d, k, largest=False)
        # b = target_r[idxs]
        # mu_local = surrogate_wtau(a, b, w[:k], smooothness_tau)
        mu_global = surrogate_wtau(pred_d, target_r, wharmonic, smooothness_tau) #/ (o ** 2 - o) * 2
        mu += mu_global
        # l = (mu_local + 1) / 2
        # g = (mu_global + 1) / 2
        # alpha = 0.5
        # mu += torch.exp((1 - alpha) * torch.log(l) + alpha * torch.log(g))

        if ref:
            tau_local += weightedtau(pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy(), rank=None)[0]
            # tau_local += weightedtau(a.cpu().detach().numpy(), b.cpu().detach().numpy(), weigher=partial(cau, gamma=gamma), rank=False)[0]
            # tau+=tau_local
            tau_global += kendalltau(pred_d.cpu().detach().numpy(), target_r.cpu().detach().numpy())[0]
            # tau += sqrt((tau_local + 1) / 2 * (tau_global + 1) / 2) * 2 - 1

    return mu / n, tau_local / n, tau_global / n
