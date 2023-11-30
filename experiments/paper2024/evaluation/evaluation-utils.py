import os
import numpy as np

DATASETS = ["bank", "cnae9", "epileptic", "fmd", "hatespeech", "imdb",
    "secom", "sentiment", "spambase", "cifar10", "coil20", "fashion_mnist",
    "har", "hiva", "orl", "seismic", "sms", "svhn"]

PROJECTION_TECHS = [
    "IDMAP",
    "PBC",
    "T-SNE",
    "UMAP",
    "SORTbmap"
]

METRICS = [
    # 'elapsed_time',
    # 'metric_pq_continuity_k_03',
    # 'metric_pq_continuity_k_05',
    # 'metric_pq_continuity_k_07',
    # 'metric_pq_continuity_k_11',
    # 'metric_pq_normalized_stress',
    # 'metric_pq_trustworthiness_k_03',
    # 'metric_pq_trustworthiness_k_05',
    # 'metric_pq_trustworthiness_k_07',
    # 'metric_pq_trustworthiness_k_11',
    # 'sortedness_cau',
    # 'sortedness_kendalltau'
    'normalized_stress',
    'trustworthiness',
    'continuity',
    'sortedness_cau',
    'sortedness_kendalltau'
]

####################################################
####################################################

##########################
# Load datasets

def load_dataset(dataset_name, datasets_folder):
    data_dir = os.path.join(os.path.expanduser('~'), datasets_folder, dataset_name)
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    return X, y