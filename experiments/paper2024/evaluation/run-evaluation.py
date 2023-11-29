import os
import numpy as np
import pandas as pd
from sortedness import sortedness
from scipy.stats import weightedtau, kendalltau
from scipy import spatial
from math import pi, sqrt
from evaluation-utils import *

##########################
## Global variables

cau = lambda r, gamma=1: 1 / pi * gamma / (gamma ** 2 + r ** 2)

# main_folder = '../data/'
datasets_folder = 'Workspace/sortedness/data/'
projections_folder = 'Workspace/sortedness/projections/'
output_folder = 'Workspace/sortedness/output/'

D_high_list = None
D_high_matrix = None
D_low_list = None
D_low_matrix = None

##########################
# Load projection

def load_projection(dataset_name, projection_name):
    # Load projection X
    proj_name = 'proj_X_' + dataset_name + '_' + projection_name + '.csv'
    proj_dir = os.path.join(os.path.expanduser('~'), projections_folder, proj_name)
    proj = pd.read_csv(proj_dir, header=None)

    # Verify
    if proj.shape[0] != X.shape[0]:
        proj = np.reshape(proj, (-1, 2))
        if proj.shape[0] != X.shape[0]:
            # proj = np.reshape(proj, (X.shape[0],-1))
            print('----------------------------------------------------')
            print("Error running: Projection returned %d rows when %d rows were expected" % (proj.shape[0], X.shape[0]))
            print('----------------------------------------------------')
            # return np.zeros((X.shape[0], 2)), y, metrics.empty_pq_metrics()
            return None

    if len(proj.shape) != 2 or proj.shape[1] != 2:
        # proj = np.reshape(proj, (X.shape[0],-1))
        print('----------------------------------------------------')
        print("Error running: Projection returned %d cols when 2 cols were expected" % (proj.shape[1]))
        print('----------------------------------------------------')
        # return np.zeros((X.shape[0], 2)), y, metrics.empty_pq_metrics()
        return None

    return proj

##########################
# Run evaluation

def run_eval(dataset_name, projection_name, X, y, proj):
    global D_low_list
    global D_low_matrix
    global D_high_list
    global D_high_matrix
    global N_SAMPLES

    N_SAMPLES = X.shape[0]

    D_low_list = spatial.distance.pdist(proj, 'euclidean')
    D_low_matrix = spatial.distance.squareform(D_low_list)

    D_high_list = spatial.distance.pdist(X, 'euclidean')
    D_high_matrix = spatial.distance.squareform(D_high_list)

    results = dict()

    results['normalized_stress'] = metric_pq_normalized_stress(D_high_matrix, D_low_matrix)
    results['trustworthiness'] = metric_trustworthiness(7, D_high_matrix, D_low_matrix)
    results['continuity'] = metric_continuity(7, D_high_matrix, D_low_matrix)
    results['sortedness_cau'] = np.mean(sortedness(X, proj, symmetric=False, weigher=cau))
    results['sortedness_kendalltau'] = np.mean(sortedness(X, proj, symmetric=False, f=kendalltau))

    output_dir = os.path.join(os.path.expanduser('~'), output_folder)
    results2dataframe(results, dataset_name).to_csv(
        '%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name), index=None)

##########################
# Normalized Stress

def metric_pq_normalized_stress(D_high, D_low):
    # D_high = np.load(DISTANCES['D_high_list'], mmap_mode='c')
    # D_low = np.load(DISTANCES[id_run]['D_low_list'], mmap_mode='c')

    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

##########################
# Trustworthiness

def metric_trustworthiness(k, D_high, D_low):
    global N_SAMPLES

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

##########################
# Continuity

def metric_continuity(k, D_high, D_low):
    global N_SAMPLES

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(N_SAMPLES):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

##########################
# results to dataframe

def results2dataframe(results, dataset_name):
    df = pd.DataFrame([results])

    return df

####################################################
####################################################

##########################
## Main

if __name__ == '__main__':
    
    for p in PROJECTION_TECHS:
        projection_name = p

        print(p)

        for d in DATASETS:
            dataset_name = d

            print(d)

            X, y = load_dataset(dataset_name, datasets_folder)
            proj = load_projection(dataset_name, projection_name)
            
            run_eval(dataset_name, projection_name, X, y, proj)

        print(" ")
