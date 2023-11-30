# This file is adapted from https://github.com/distillpub/post--misread-tsne
# The original file is licensed under the Apache 2.0 License.
# This file is also licensed under the Apache 2.0 License.

import math
import os
from colorsys import hls_to_rgb

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.manifold import TSNE

from sortedness.embedding import balanced_embedding

np.random.seed(0)
alpha = 0
beta = 0.5
kappa = 5
activation_function = "relu"
neurons = 100
lmbd = 0.01
show = False
# show = True


## A point with color info.
class Point:
    def __init__(self, coords, color='#039'):
        self.coords = coords
        self.color = color


## Euclidean distance.
def dist(a, b):
    d = 0
    for i, val in enumerate(a):
        d += (val - b[i]) ** 2
    return math.sqrt(d)


## Gaussian generator, mean = 0, std = 1.
def normal():
    return norm.rvs(0, 1)


## Create random Gaussian vector.
def normal_vector(dim):
    return norm.rvs(0, 1, size=dim)


## Scale the given vector.
def scale(vector, a):
    for i, val in enumerate(vector):
        vector[i] *= a


## Add two vectors.
def add(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result


## Adds colors to points depending on 2D location of original.
def add_spatial_colors(points):
    x_extent = np.min(points, axis=0)[0], np.max(points, axis=0)[0]
    y_extent = np.min(points, axis=0)[1], np.max(points, axis=0)[1]
    x_scale = np.linspace(x_extent[0], x_extent[1], 256)[:, np.newaxis]
    y_scale = np.linspace(y_extent[0], y_extent[1], 256)
    for point in points:
        x, y = point.coords
        c1 = int(np.interp(x, x_extent, np.arange(256)))
        c2 = int(np.interp(y, y_extent, np.arange(256)))
        point.color = f'rgb(20, {c1}, {c2})'


## Convenience function to wrap 2d arrays as Points, using a default color scheme.
def make_points(originals, color_scheme='viridis'):
    points = [Point(coords) for coords in originals]
    add_spatial_colors(points)
    return points


## Creates distance matrix for t-SNE input.
def distance_matrix(points):
    n = len(points)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = dist(points[i].coords, points[j].coords)

    return matrix


## Data in shape of 2D grid.
def grid_data(size):
    print("grid_data")
    points = []
    for x in range(size):
        for y in range(size):
            points.append([x, y])

    return make_points(points)


## Gaussian cloud, symmetric, of given dimension.
def gaussian_data(n, dim=2):
    print("gaussian_data")
    points = []
    for _ in range(n):
        p = np.random.normal(size=dim)
        points.append(Point(p))
    return points


## Elongated Gaussian ellipsoid.
def elongated_gaussian_data(n, dim=2):
    print("elongated_gaussian_data")
    points = []
    for _ in range(n):
        p = np.random.normal(size=dim)
        p /= (1 + np.arange(dim))
        points.append(Point(p))
    return points


## Return a color for the given angle.
def angle_color(t):
    hue = (300 * t) / (2 * math.pi)
    return hls_to_rgb(hue, 0.5, 0.5)


## Data in a 2D circle, regularly spaced.
def circle_data(num_points):
    print("circle_data")
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        coords = [np.cos(angle), np.sin(angle)]
        color = hls_to_rgb(angle / (2 * np.pi), 0.5, 0.5)
        points.append(Point(coords, color))

    return points


## Random points on a 2D circle.
def random_circle_data(num_points):
    print("random_circle_data")
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * np.random.random()
        coords = [np.cos(angle), np.sin(angle)]
        color = hls_to_rgb(angle / (2 * np.pi), 0.5, 0.5)
        points.append(Point(coords, color))

    return points


## Clusters arranged in a circle.
def random_circle_cluster_data(num_points):
    print("random_circle_cluster_data")
    num_points //= 2
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        color = hls_to_rgb(angle / (2 * np.pi), 0.5, 0.5)
        for _ in range(20):
            x = np.cos(angle) + 0.01 * np.random.normal()
            y = np.sin(angle) + 0.01 * np.random.normal()
            points.append(Point([x, y], color))

    return points


## Two 2D clusters of different sizes.
def two_different_clusters_data_2d(n):
    print("two_different_clusters_data_2d")
    points = []
    for i in range(n):
        points.append(Point([10 * np.random.normal(), 10 * np.random.normal()], '#039'))
        points.append(Point([100 + np.random.normal(), np.random.normal()], '#f90'))
    return points


## Two clusters of the same size.
def two_clusters_data(n, dim=50):
    print("two_clusters_data")
    points = []
    for _ in range(n):
        points.append(Point(np.random.normal(size=dim), '#039'))
        v = np.random.normal(size=dim)
        v[0] += 10
        points.append(Point(v, '#f90'))
    return points


## Two differently sized clusters, of arbitrary dimensions.
def two_different_clusters_data(n, dim=50, scale=10):
    print("two_different_clusters_data")
    dim = dim or 50
    scale = scale or 10
    points = []
    for _ in range(n):
        points.append(Point(np.random.normal(size=dim), '#039'))
        v = np.random.normal(size=dim)
        v /= scale
        v[0] += 20
        points.append(Point(v, '#f90'))
    return points


## Three clusters, at different distances from each other, in 2D
def three_clusters_data_2d(n):
    print("three_clusters_data_2d")
    points = []
    for _ in range(n):
        points.append(Point([np.random.normal(), np.random.normal()], '#039'))
        points.append(Point([10 + np.random.normal(), np.random.normal()], '#f90'))
        points.append(Point([50 + np.random.normal(), np.random.normal()], '#6a3'))
    return points


## Three clusters, at different distances from each other, in any dimension.
def three_clusters_data(n, dim=50):
    print("three_clusters_data")
    dim = dim or 50
    points = []
    for _ in range(n):
        p1 = np.random.normal(size=dim)
        points.append(Point(p1, '#039'))
        p2 = np.random.normal(size=dim)
        p2[0] += 10
        points.append(Point(p2, '#f90'))
        p3 = np.random.normal(size=dim)
        p3[0] += 50
        points.append(Point(p3, '#6a3'))
    return points


## One tiny cluster inside of a big cluster.
def subset_clusters_data(n, dim=2):
    print("subset_clusters_data")
    dim = dim or 2
    points = []
    for _ in range(n):
        p1 = np.random.normal(size=dim)
        points.append(Point(p1, '#039'))
        p2 = np.random.normal(size=dim)
        p2 *= 50
        points.append(Point(p2, '#f90'))
    return points


## Data in a rough simplex.
def simplex_data(n, noise=0):
    print("simplex_data")
    points = []
    for i in range(n):
        p = []
        for j in range(n):
            value = 1 if i == j else 0
            noise_factor = noise * np.random.normal()
            p.append(value + noise_factor)
        points.append(Point(p))

    return points


## Uniform points from a cube.
def cube_data(n, dim=2):
    print("cube_data")
    points = []
    for _ in range(n):
        p = np.random.uniform(size=dim)
        points.append(Point(p))

    return points


## Points in two unlinked rings.
def unlink_data(n):
    print("unlink_data")
    points = []

    def rotate(x, y, z):
        u = x
        cos = np.cos(0.4)
        sin = np.sin(0.4)
        v = cos * y + sin * z
        w = -sin * y + cos * z
        return [u, v, w]

    for i in range(n):
        t = 2 * math.pi * i / n
        sin = math.sin(t)
        cos = math.cos(t)

        # Ring 1
        points.append(Point(rotate(cos, sin, 0), '#f90'))

        # Ring 2
        points.append(Point(rotate(3 + cos, 0, sin), '#039'))

    return points


## Points in linked rings.
def link_data(n):
    print("link_data")
    points = []

    def rotate(x, y, z):
        u = x
        cos = np.cos(0.4)
        sin = np.sin(0.4)
        v = cos * y + sin * z
        w = -sin * y + cos * z
        return [u, v, w]

    for i in range(n):
        t = 2 * math.pi * i / n
        sin = math.sin(t)
        cos = math.cos(t)

        # Ring 1
        points.append(Point(rotate(cos, sin, 0), '#f90'))

        # Ring 2
        points.append(Point(rotate(1.1 + cos, sin, 0), '#039'))

    return points


## Points in a trefoil knot.
def trefoil_data(n):
    print("trefoil_data")
    points = []
    for i in range(n):
        t = 2 * np.pi * i / n
        x = np.sin(t) + 2 * np.sin(2 * t)
        y = np.cos(t) - 2 * np.cos(2 * t)
        z = -np.sin(3 * t)
        color = hls_to_rgb(t / (2 * np.pi), 0.5, 0.5)
        points.append(Point([x, y, z], color))
        # points.append(Point([x, y, z], angle_color(t)))

    return points


def spiral(n):
    a = .11
    phi = np.arange(0, 100 * np.pi, 0.39)
    x = a * phi * np.cos(phi)
    y = a * phi * np.sin(phi)
    return np.array(list(zip(x, y)))[:n]


## Two long, linear clusters in 2D.
def long_cluster_data(n):
    print("long_cluster_data")
    points = []
    s = 0.03 * n
    for i in range(n):
        x1 = i + s * np.random.normal()
        y1 = i + s * np.random.normal()
        points.append(Point([x1, y1], '#039'))
        x2 = i + s * np.random.normal() + n / 5
        y2 = i + s * np.random.normal() - n / 5
        points.append(Point([x2, y2], '#f90'))
    return points


## Mutually orthogonal steps.
def ortho_curve(n):
    print("ortho_curve")
    points = []
    for i in range(n):
        coords = np.zeros(n)
        coords[i] = 1
        t = 1.5 * np.pi * i / n
        color = hls_to_rgb(t / (2 * np.pi), 0.5, 0.5)
        points.append(Point(coords, color))
    return points


## Random walk
def random_walk(n, dim=2):
    print("random_walk")
    points = []
    current = np.zeros(dim)

    for i in range(n):
        step = np.random.normal(size=dim)
        next_point = current + step
        color = hls_to_rgb(1.5 * np.pi * i / n, 0.5, 0.5)
        points.append(Point(next_point, color))
        current = next_point

    return points


## Random walk
def random_jump(n, dim=2):
    print("random_jump")
    points = []
    current = np.zeros(dim)

    for i in range(n):
        step = np.random.normal(size=dim)
        next_point = current + step
        jump_vector = np.random.normal(size=dim)
        jump_vector_scaled = jump_vector * np.sqrt(dim)
        next_point_with_jump = next_point + jump_vector_scaled
        color = hls_to_rgb(1.5 * np.pi * i / n, 0.5, 0.5)
        points.append(Point(next_point_with_jump, color))
        current = next_point

    return points


##########################################
##########################################

def getCoords(points):
    coords = []
    for point in points:
        coords.append(point.coords)
    return coords


def getColors(points):
    colors = []
    for point in points:
        colors.append(point.color)
    return colors


def plot_proj(X, y):
    plt.scatter(X[:, 0], X[:, 1], color=y)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_proj_r(X, y, r):
    plt.scatter(X[:, 0], X[:, 1], color=y, s=r)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_all_r(*args, y, exp, r):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)

    for ax, X, name in zip(axes, args, ["Original", "t-SNE", "SORTBmap"]):

        if X.shape[1] > 2:
            index = list(range(X.shape[0]))
            index.sort(key=lambda i: X[i, 2])

            # Usando o loop for
            y_new = y.copy()
            r_new = r.copy()
            a = 0
            for i in index:
                y_new[a] = y[i]
                r_new[a] = r[i]
                a = a + 1

        min_vals = np.min(X[:, :2], axis=0, keepdims=True)
        max_vals = np.max(X[:, :2], axis=0, keepdims=True)
        X = (X[:, :2] - min_vals) / (max_vals - min_vals)

        if True or name == "Original":
            if r:
                ax.scatter(X[index, 0], X[index, 1], color=y_new, s=r_new, edgecolor="white")
            else:
                ax.scatter(X[:, 0], X[:, 1], color=y)
        else:
            ax.scatter(X[:, 0], X[:, 1], color=y)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    if show:
        plt.show()
        return

    # output_folder = 'Documentos/eurovis2024/'
    output_folder = 'pCloudDrive/Pesquisa/sortedness/eurovis2024'
    output_dir = os.path.join(os.path.expanduser('~'), output_folder)
    plt.savefig("%s/Distill_Figue_%s_a%s.pdf" % (output_dir, exp, alpha), format='pdf')
    plt.savefig("%s/Distill_Figue_%s_a%s.png" % (output_dir, exp, alpha), format='png')


def plot_all(*args, y, exp):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)

    for ax, X, name in zip(axes, args, ["Original", "t-SNE", "SORTBmap"]):
        # ax.set_xlim([-0.05, 1.05])
        # ax.set_ylim([-0.05, 1.05])

        min_vals = np.min(X[:, :2], axis=0, keepdims=True)
        max_vals = np.max(X[:, :2], axis=0, keepdims=True)
        X = (X[:, :2] - min_vals) / (max_vals - min_vals)

        ax.scatter(X[:, 0], X[:, 1], color=y)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    if show:
        plt.show()
        exit()

    # output_folder = 'Documentos/eurovis2024/'
    output_folder = 'pCloudDrive/Pesquisa/sortedness/eurovis2024'
    output_dir = os.path.join(os.path.expanduser('~'), output_folder)
    plt.savefig("%s/Distill_Figue_%s_a%s.pdf" % (output_dir, exp, alpha), format='pdf')
    plt.savefig("%s/Distill_Figue_%s_a%s.png" % (output_dir, exp, alpha), format='png')


def getRadius(points):
    # Extract coordinates from points
    coordinates = np.array(getCoords(points))

    # Calculate extents in each dimension
    x_extent = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
    y_extent = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    z_extent = np.min(coordinates[:, 2]), np.max(coordinates[:, 2])
    z_scale = plt.cm.ScalarMappable(cmap="viridis")
    z_scale.set_clim(z_extent[0], z_extent[1])

    # Calculate center of data points
    center_x = (x_extent[0] + x_extent[1]) / 2
    center_y = (y_extent[0] + y_extent[1]) / 2

    # Calculate scale based on extents and canvas size
    scale = min(100, 100) / max(x_extent[1] - x_extent[0], y_extent[1] - y_extent[0])
    scale *= 0.9

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    r = []

    # Plot circles for each point
    for point in points:
        color = point.color

        x = (point.coords[0] - center_x) * scale + 250
        y = -(point.coords[1] - center_y) * scale + 350

        radius = z_scale.to_rgba(point.coords[2])[0] * 10

        r.append(radius)

    return r


##########################################
##########################################
# Distill study cases

## Cluster sizes in a t-SNE plot mean nothing
def case02(p=50, n=5000, e=100):
    points = two_different_clusters_data_2d(75)

    X = np.array(getCoords(points))
    y = getColors(points)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp=2)


## Distances between clusters might not mean anything - 50 points
def case03a(p=50, n=5000, e=100):
    points = three_clusters_data_2d(50)

    X = np.array(getCoords(points))
    y = getColors(points)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp='3a')


## Distances between clusters might not mean anything - 200 points
def case03b(p=50, n=5000, e=100):
    points = three_clusters_data_2d(200)

    X = np.array(getCoords(points))
    y = getColors(points)

    # plot_proj(X, y)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp='3b')


## Random noise doesn’t always look random - 500 points
def case04(p=50, n=5000, e=100):
    points = gaussian_data(500, 100)

    X = np.array(getCoords(points))
    y = getColors(points)

    # plot_proj(X, y)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp='4')


## You can see some shapes, sometimes
def case05a(p=50, n=5000, e=100):
    points = long_cluster_data(75)

    X = np.array(getCoords(points))
    y = getColors(points)

    # plot_proj(X, y)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp='5a')


## For topology, you may need more than one plot
def case06a(p=50, n=5000, e=100):
    points = subset_clusters_data(75, 50)

    X = np.array(getCoords(points))
    y = getColors(points)

    # plot_proj(X, y)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    plot_all(X, tsne_proj, sort_proj, y=y, exp='6a')


## For topology, you may need more than one plot
def case06b(p=50, n=5000, e=100):
    points = unlink_data(75)

    X = np.array(getCoords(points))
    y = getColors(points)

    r = getRadius(points)
    r = [i * 50 for i in r]
    # plot_proj_r(X, y, r)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    # plot_all(X, tsne_proj, sort_proj, y=y, exp='6b')
    plot_all_r(X, tsne_proj, sort_proj, y=y, exp='6b', r=r)


## For topology, you may need more than one plot
def case06c(p=50, n=5000, e=100):
    points = trefoil_data(75)

    X = np.array(getCoords(points))
    y = getColors(points)

    r = getRadius(points)
    r = [i * 50 for i in r]
    # plot_proj_r(X, y, r)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    # plot_all(X, tsne_proj, sort_proj, y=y, exp='6c')
    plot_all_r(X, tsne_proj, sort_proj, y=y, exp='6c', r=r)


def case_spiral(p=50, n=5000, e=100):
    points = spiral(75)

    X = np.array(getCoords(points))
    y = getColors(points)

    r = getRadius(points)
    r = [i * 50 for i in r]
    # plot_proj_r(X, y, r)

    print("tsne...")
    tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=p
                     , n_iter=n, n_jobs=-1).fit_transform(X)

    print("sort...")
    sort_proj = balanced_embedding(X, alpha=alpha, beta=beta, hidden_layers=[neurons], kappa=kappa, epochs=e, activation_functions=[activation_function], smoothness_tau=lmbd
                                   , return_only_X_=True)

    # plot_all(X, tsne_proj, sort_proj, y=y, exp='6c')
    plot_all_r(X, tsne_proj, sort_proj, y=y, exp='6c', r=r)


## Main
if __name__ == '__main__':
    tsne_perplexity = 50
    tsne_n_iter = 2000
    sort_epochs = 2000

    # tsne_n_iter = 250
    # sort_epochs = 10

    case06c(tsne_perplexity, tsne_n_iter, sort_epochs)
    # exit()
    case03b(tsne_perplexity, tsne_n_iter, sort_epochs)
    case02(tsne_perplexity, tsne_n_iter, sort_epochs)
    case04(tsne_perplexity, tsne_n_iter, sort_epochs)

    case03a(tsne_perplexity, tsne_n_iter, sort_epochs)


    case05a(tsne_perplexity, tsne_n_iter, sort_epochs)

    case06a(tsne_perplexity, tsne_n_iter, sort_epochs)
    case06b(tsne_perplexity, tsne_n_iter, sort_epochs)

