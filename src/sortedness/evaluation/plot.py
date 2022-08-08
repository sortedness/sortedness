#  Copyright (c) 2022. Davi Pereira dos Santos
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

from dataclasses import dataclass

from hoshmap import idict

colors = ["blue", "green", "red", "gray", "yellow", "black", "pink"]


@dataclass
class Plot:
    d: idict
    projection: str
    xlabel: str
    ylabel: str
    legend: bool
    plt: callable
    fontsize: int = 18
    marksize: int = 2

    def __post_init__(self):
        self.fig = self.plt.figure()
        self.plt.title(f"{self.projection}", fontsize=self.fontsize)
        self.ax = self.fig.gca()
        # ax.set_yscale('log')

    def __lshift__(self, other):
        slabel, color = other
        self.ax.scatter(self.d[self.xlabel], self.d[slabel], s=self.marksize, c=color, label=slabel)

    def finish(self):
        if self.legend:
            self.ax.legend(fontsize=self.fontsize)
        self.plt.xlabel(self.xlabel, fontsize=self.fontsize)
        self.plt.ylabel(self.ylabel, fontsize=self.fontsize)
        self.plt.tight_layout()
        figManager = self.plt.get_current_fig_manager()
        figManager.window.showMaximized()
        figManager.window.setWindowTitle(f"{self.projection}     {self.xlabel}")
