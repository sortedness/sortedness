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

from math import dist

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from sortedness.local import stress, sortedness, pwsortedness

np.random.seed(86)
fig, ax = plt.subplots()
points = 100
x, y = np.random.uniform(0, 10, size=points).tolist(), np.random.uniform(0, 10, size=points).tolist()
sc = ax.scatter(x, y, s=300, c=np.random.uniform(0, 1, points))
plt.xlim(0, 10)
plt.ylim(0, 10)
o = DataFrame(list(zip(x, y)))

print(o)
p = o.copy()
selected = [None] * 3
print(1111111111111, len(selected))


def f(idx):
    # plt.title(f"σ - 1: {(round(1 - mean(stress(o, p)), 2))}"
    #           f"        gλτ: {round(mean(global_gtau(o, p)), 2)}"
    #           f"        λτw: {round(mean(sortedness(o, p)), 2)}"
    #           f"        Λτw: {round(mean(pwsortedness(o, p)), 2)}", fontsize=30)
    plt.title(
        f"σ - 1: {round(1 - stress(o, p)[idx], 2)}"
              # f"        gτw: {round(gtau(idx, o, p), 2)}"
              f"        λτw: {round(sortedness(o, p, idx, weigher=lambda r:1/(1+10*r/points)), 2)}"
              f"        Λτw: {round(pwsortedness(o, p, idx), 2)}", fontsize=30)


def onmove(event):
    x, y = event.xdata, event.ydata
    if None in [x, y] or selected[2] is None:
        return
    p.iloc[selected[2]] = [x, y]
    f(selected[2])

    n = p.to_numpy()
    sc.set_offsets(np.c_[n[:, 0].tolist(), n[:, 1].tolist()])
    fig.canvas.draw_idle()


def onpress(event):
    # event.button
    x, y = event.xdata, event.ydata
    dmin = 99999
    for i, a in enumerate(p.to_numpy()):
        if (d := dist([x, y], a)) < dmin:
            dmin = d
            idx = i
    print(idx, dmin)
    r = p.iloc[idx].to_numpy().tolist()
    selected[0] = r[0]
    selected[1] = r[1]
    selected[2] = idx
    f(selected[2])


def onrelease(event):
    x, y = event.xdata, event.ydata
    p.iloc[selected[2]] = [x, y]
    f(selected[2])
    fig.canvas.draw_idle()
    selected[2] = None


fig.canvas.mpl_connect("motion_notify_event", onmove)
fig.canvas.mpl_connect('button_press_event', onpress)
fig.canvas.mpl_connect('button_release_event', onrelease)

plt.show()
