#  Copyright (c) 2024. Davi Pereira dos Santos
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

from sortedness.new.dataset import mnist
from sortedness.new.quality import *
from sortedness.new.sktransformer import SKTransformer
from sortedness.new.weighting import gaussian, cauchy

n = 300
X, colors = mnist(n)
n = len(X)
labels = colors[:100]
gauw = gaussian(17, kappa=5)
cauw = cauchy(20, kappa=5)
# todo: fazer cada função de ponderação retornar apenas os k vizinhos mais relevantes conforme abaixo...
#  epsilon=0.00001
#  k = int(halfnorm.ppf(1 - epsilon, 0, sigma))

stress, transi, sort = Calmness(X), Transitiveness(X), Sortedness(X)
wstress, wtransi, wsort = Calmness(X, cauw), Transitiveness(X, cauw), Sortedness(X, gauw)

# # noinspection PyTypeChecker
# c = SKTransformer(3 / (1 / Calmness(X, ) + 1 / Transitiveness(X, ) + 1 / Sortedness(X, gauw)), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# # noinspection PyTypeChecker
# c = SKTransformer((stress + gtrans + lsort) / 3, verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# noinspection PyTypeChecker
# c = SKTransformer((wstress)**2, verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# c = SKTransformer((Transitiveness(X, cauw, lambd=1)+Transitiveness(X, cauw, lambd=1, sortbyX_=False))/2, verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# c = SKTransformer(2/(1/Calmness(X, ) +1/ Transitiveness(X, cauw)), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# c = SKTransformer(Calmness(X, ) * Sortedness(X, gauw), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()
#
c = SKTransformer(Sortedness(X, gauw), verbose=True)
c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
exit()

# c = SKTransformer(RelativeCalmness(X, cauw), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# c = SKTransformer(Linearity(X), batch_size=X.shape[0], verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()
#
# c = SKTransformer(KLConcordanceGaussianCauchy(X), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

# c = SKTransformer(Transitiveness(X, cauw), verbose=True)
# c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
# exit()

c = SKTransformer(Calmness(X, cauw), verbose=True)
c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
exit()

# ann = M(X, d=2, hidden_layers=[20, 10], activation_functions=["tanh", "relu"])
# hyperoptimizer = gdtuo.RMSProp(optimizer=gdtuo.SGD(alpha=0.01, mu=0.0))
# hyperoptimizer = gdtuo.Adam(optimizer=gdtuo.SGD(alpha=0.01, mu=0.0))

# TODO: quando coloca pesos em X fica muito ruim → checar implementação; será que perde gradiente por reindexar D_ ordenado por D? sempre trabalha nos mesmos vizinhos
# TODO: transitiveness ponderado+soft está com valores muito baixos, mesmo com bom plot visual
