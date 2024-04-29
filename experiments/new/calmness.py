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
from sortedness.new.quality import Calmness
from sortedness.new.sktransformer import SKTransformer
from sortedness.new.weighting import cauchy

n = 500
X, colors = mnist(n)
labels = colors[:50]
# w = gaussian(n)
w = cauchy(n//2, kappa=10)
# todo: fazer cada função de ponderação retornar apenas os k vizinhos mais relevantes conforme abaixo...
#  epsilon=0.00001
#  k = int(halfnorm.ppf(1 - epsilon, 0, sigma))


c = SKTransformer(Calmness(X, w), verbose=True)

c.fit(X, plot=True, plot_labels=labels, plot_colors=colors)
