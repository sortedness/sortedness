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
import torch

from sortedness.new.quality._calmness import Calmness
from sortedness.new.quality._klconcordance import KLConcordanceGaussianCauchy
from sortedness.new.quality._linearity import Linearity
from sortedness.new.quality._relativecalmness import RelativeCalmness
from sortedness.new.quality._sortedness import Sortedness
from sortedness.new.quality._transitiveness import Transitiveness

_ = Calmness
_ = RelativeCalmness
_ = Sortedness
_ = Transitiveness
_ = KLConcordanceGaussianCauchy
_ = Linearity

pdist = torch.nn.PairwiseDistance(p=2, keepdim=True, eps=0.00000001)
