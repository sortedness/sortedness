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
import matplotlib.pyplot as plt
import numpy as np


# Function to create a gradient along the rows
def create_gradient(n):
    return np.linspace(0, 1, n)


# Function to create a 5x5 grid with alternating colors in columns
def create_colored_grid():
    grid = np.zeros((5, 5, 3))

    # Define three alternating colors for the columns
    colors = np.array([[0.2, 0.2, 0.8], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2]])

    g=1.05
    for i in range(5):
        grid[:, i, :] = colors[i % 3]*g % 1
        g*=g

    return grid


# Create a 25x25 grid of numbers
grid_size = 25
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xticks(np.arange(0, grid_size, 5))
ax.set_yticks(np.arange(0, grid_size, 5))
ax.set_xticklabels([])
ax.set_yticklabels([])

# Populate the scatterplot with alternating colors in columns and gradient along rows
g=1.01
for i in range(0, grid_size, 5):
    g*=g
    for j in range(0, grid_size, 5):
        gradient = create_gradient(5)
        colored_grid = create_colored_grid()

        for k in range(5):
            ax.scatter(
                np.arange(j, j + 5),
                np.ones(5) * (i + k),
                c=(colored_grid[:, k, :]*g) % 1,
                cmap='viridis',
                s=100,
            )

plt.show()
