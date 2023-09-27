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
from matplotlib.backend_bases import MouseButton
from pandas import DataFrame
from scipy.stats import kendalltau

from sortedness.local import stress, sortedness

np.random.seed(86)
ax = [0]
fig, ax[0] = plt.subplots()
points = 2000
x, y = np.random.uniform(0, 10, size=points).tolist(), np.random.uniform(0, 10, size=points).tolist()
red = (1, 0, 0, 0.9)
blue = (0, 0, 1, 0.9)
gray = (0.3, 0.4, 0.5, 0.3)
colors = [red] + [gray] * (points - 1)
sc = [0]
sc[0] = ax[0].scatter(x, y, s=200, c=colors)
plt.xlim(0, 10)
plt.ylim(0, 10)
o = DataFrame(list(zip(x, y)))
o.iloc[0] = -1, -1
print(o)
p = o.copy()
selected = dict(coords=None, idx=0, old_color=gray)
drag = dict(coords=None, idx=None, old_color=gray, old_idx=None)


def f():
    idx = selected["idx"]
    plt.title(
        f"σ - 1: {np.round(1 - stress(o, p, idx), 5)}"
        f"      λτ1: {np.round(sortedness(o, p, idx, f=kendalltau), 5)}"
        f"      λτw: {np.round(sortedness(o, p, idx), 5)}"
        # f"      Λτw: {round(pwsortedness(o, p, i=idx), 5)}"
        , fontsize=26)


def onmove(event, keepstill=False):
    x, y = event.xdata, event.ydata
    if None in [x, y] or drag["idx"] is None:
        return
    p.iloc[drag["idx"]] = [x, y] - drag["delta"]
    f()

    n = p.to_numpy()
    sc[0].set_offsets(np.c_[n[:, 0].tolist(), n[:, 1].tolist()])
    fig.canvas.draw_idle()
    if not keepstill:
        if selected["idx"] == drag["idx"]:
            selected["old_color"] = blue
        else:
            drag["old_color"] = blue


def onpress(event):
    x, y = event.xdata, event.ydata
    dmin = 99999
    for i, a in enumerate(p.to_numpy()):
        if (d := dist([x, y], a)) < dmin:
            dmin = d
            idx = i
    drag["coords"] = p.iloc[idx].to_numpy()
    drag["delta"] = [x, y] - drag["coords"]

    if event.button == MouseButton.RIGHT:
        colors[selected["idx"]] = selected["old_color"]
        selected["old_color"] = drag["old_color"] if idx == drag["old_idx"] else colors[idx]
        colors[idx] = red
        selected["idx"] = idx
    elif colors[idx] != red:
        if drag["old_idx"] is not None:
            colors[drag["old_idx"]] = drag["old_color"]
        drag["old_color"] = colors[idx]
        colors[idx] = blue
        drag["old_idx"] = idx
    drag["idx"] = idx
    ax[0].cla()
    sc[0] = ax[0].scatter(p[0], p[1], s=200, c=colors)
    onmove(event, keepstill=True)


def onrelease(event):
    drag["idx"] = None


def onkeyrelease(event):
    if event.key == "z":
        p.iloc[drag["idx"]] = drag["coords"]
        f()
        fig.canvas.draw_idle()
        n = p.to_numpy()
        sc[0].set_offsets(np.c_[n[:, 0].tolist(), n[:, 1].tolist()])
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", onmove)
fig.canvas.mpl_connect('button_press_event', onpress)
fig.canvas.mpl_connect('button_release_event', onrelease)
fig.canvas.mpl_connect('key_release_event', onkeyrelease)

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
ax[0].cla()
sc[0] = ax[0].scatter(p[0], p[1], s=200, c=colors)
n = p.to_numpy()
sc[0].set_offsets(np.c_[n[:, 0].tolist(), n[:, 1].tolist()])
plt.show()
