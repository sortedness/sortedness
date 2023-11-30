import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation-utils import *

def load_values(dataset_name, projection_name, output_dir):
    file_name = ('%s/%s_%s_pq_results.csv' % (output_dir, dataset_name, projection_name))
    df = pd.read_csv(file_name, header=None)

    return df.values[1,:]

def get_metric_values(results, index):    
    return(results[:, index])

def plot_boxplot(tech_values, i, output_dir):
    metric = metrics[i]

    print(metric)

    m = []

    for t in projection_techs:
        data = get_metric_values(tech_values[t], i)
        for d in data:
            v = [d, t]
            m.append(v)

    df = pd.DataFrame(m)
    df.columns = [metric, "tech"]
    print(df.columns)
    df[metric] = pd.to_numeric(df[metric])
    df["tech"] = df["tech"].astype(str)
    
    plt.clf()
    sns.boxplot( x=df['tech'], y=df[metric], color=sns.color_palette("Set2")[i], showfliers = False)

    # save the plot
    # plt.savefig("seaborn_plot.svg", format='svg')
    plt.savefig("%s/plots/%s_plot.pdf"  % (output_dir, metric), format='pdf')
    plt.savefig("%s/plots/%s_plot.png"  % (output_dir, metric), format='png')
    # sns.plt.savefig('ax.png')


####################################################
####################################################

##########################
## Main

if __name__ == '__main__':
    output_folder = 'Workspace/sortedness/output/'
    output_dir = os.path.join(os.path.expanduser('~'), output_folder)
    if not os.path.exists(output_dir):
        print('Directory %s not found' % output_dir)
        exit(1)

    ########################
    ## Build values matrix 

    tech_values = dict()

    for p in PROJECTION_TECHS:
        projection_name = p

        # results = np.ndarray((0, 12))
        results = np.ndarray((0, 5))
        
        for d in DATASETS:
            dataset_name = d

            v = load_values(dataset_name, projection_name, output_dir)

            results = np.append(results, [v], axis=0)

        tech_values[projection_name] = results

    ###########
    ## Plots

    for i in range(0,5):
        plot_boxplot(tech_values, i, output_dir)