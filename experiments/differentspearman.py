# #  Copyright (c) 2023. Davi Pereira dos Santos
# #  This file is part of the sortedness project.
# #  Please respect the license - more about this in the section (*) below.
# #
# #  sortedness is free software: you can redistribute it and/or modify
# #  it under the terms of the GNU General Public License as published by
# #  the Free Software Foundation, either version 3 of the License, or
# #  (at your option) any later version.
# #
# #  sortedness is distributed in the hope that it will be useful,
# #  but WITHOUT ANY WARRANTY; without even the implied warranty of
# #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #  GNU General Public License for more details.
# #
# #  You should have received a copy of the GNU General Public License
# #  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
# #
# #  (*) Removing authorship by any means, e.g. by distribution of derived
# #  works or verbatim, obfuscated, compiled or rewritten versions of any
# #  part of this work is illegal and it is unethical regarding the effort and
# #  time spent here.
# #
#
# import torch
# # import torchsort
#
#
# def spearmanr(pred, target, **kw):
#     pred = torchsort.soft_rank(pred, **kw)
#     target = torchsort.soft_rank(target, **kw)
#     pred = pred - pred.mean()
#     pred = pred / pred.norm()
#     target = target - target.mean()
#     target = target / target.norm()
#     return (pred * target).sum()
#
#
# spearman1 = spearmanr(torch.tensor([[1., 2., 3., 4., 5, 6, 6.2, 8, 9, 10.]]), torch.tensor([[1., 2., 3., 4., 5, 6, 6.2, 8, 9, 10.]]))
# spearman2 = spearmanr(torch.tensor([[1., 2., 3., 4., 5, 6, 6.2, 8, 9, 10.]]), torch.tensor([[1., 2., 3., 4., 5, 6, 7.8, 8, 9, 10.]]))
# print(float(spearman1), float(spearman2))
#
# # todo: artigo: differentiability detecta mudança:    A B   C  →  A   B C     sem recorrer a pairwise.
