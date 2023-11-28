# This file is adapted from https://github.com/distillpub/post--misread-tsne
# The original file is licensed under the Apache 2.0 License.
# This file is also licensed under the Apache 2.0 License.

import math
from colorsys import hls_to_rgb

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.manifold import TSNE

from sortedness.embedding import balanced_embedding


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
    points = []
    for x in range(size):
        for y in range(size):
            points.append([x, y])

    return make_points(points)


## Gaussian cloud, symmetric, of given dimension.
def gaussian_data(n, dim=2):
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

    return points


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


def plot_points(points):
    # matplotlib.use('QtAgg')
    for point in points:
        plt.scatter(point.coords[0], point.coords[1], color=point.color)
    plt.show()

    # output_folder = 'Documentos/eurovis2024/'
    # output_dir = os.path.join(os.path.expanduser('~'), output_folder)
    # plt.savefig("%s/plots/%s_plot.pdf"  % (output_dir, metric), format='pdf')
    # plt.savefig("%s/plots/%s_plot.png"  % (output_dir, metric), format='png')
    # plt.savefig("%s/plots/fig1.pdf"  % (output_dir, metric), format='pdf')
    # plt.savefig("%s/plots/fig1.png"  % (output_dir, metric), format='png')


def plot_proj(X, y):
    plt.scatter(X[:, 0], X[:, 1], color=y)
    plt.show()


def plot_all(*args, y):
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(12, 4)
    for ax, X, name in zip(axes, args, ["Original", "t-SNE", "SORTbmap"]):
        # ax.set_xlim([-0.05, 1.05])
        # ax.set_ylim([-0.05, 1.05])

        min_vals = np.min(X[:, :2], axis=0, keepdims=True)
        max_vals = np.max(X[:, :2], axis=0, keepdims=True)
        X = (X[:, :2] - min_vals) / (max_vals - min_vals)

        ax.scatter(X[:, 0], X[:, 1], color=y)
        ax.set_title(name)
    plt.show()


## Main
if __name__ == '__main__':
    # for f in [two_clusters_data, gaussian_data, elongated_gaussian_data, circle_data, random_circle_data, random_circle_cluster_data, two_different_clusters_data_2d, two_clusters_data, two_different_clusters_data, three_clusters_data_2d, three_clusters_data, subset_clusters_data, simplex_data, cube_data, unlink_data, link_data, trefoil_data, long_cluster_data, ortho_curve, random_walk, random_jump]:
    for f in [two_different_clusters_data_2d, two_clusters_data, two_different_clusters_data, three_clusters_data_2d, random_circle_cluster_data, two_clusters_data, gaussian_data, elongated_gaussian_data, circle_data, random_circle_data, three_clusters_data, subset_clusters_data, simplex_data, cube_data, unlink_data, link_data, trefoil_data, long_cluster_data, ortho_curve, random_walk, random_jump]:
        points = f(75)

        # plot_points(points)

        X = np.array(getCoords(points))
        y = getColors(points)
        sort_proj = balanced_embedding(X, alpha=1, epochs=25, activation_functions=["relu"], return_only_X_=True)

        # plot_proj(X_, y)

        tsne_proj = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=300, n_jobs=-1).fit_transform(X)

        # plot_proj(xcp, y)

        plot_all(X, tsne_proj, sort_proj, y=y)

        # for p in points:
        #     print(p.coords)
