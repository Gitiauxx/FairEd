import matplotlib.pyplot as plt
import pandas as pd

import os
import numpy as np

def plot_fairness_metric_baseline(datafile, outfolder):

    data = pd.read_csv(datafile, index_col='classifier')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0']

    width = 0.35
    labels = ['DemPY', 'DisIP', 'EqOPP']
    x = np.arange(len(labels))

    for i, name in enumerate(list(data.index)):
        ax[0].bar(x + i * width / len(data), data.loc[name, ['demDP_gender', 'disIP_gender', 'eqOPP_gender']], width / len(data), color=colors[i])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=10)
    ax[0].set_title('Fairness Metrics across Gender', size=14)

    for i, name in enumerate(list(data.index)):
        ax[1].bar(x + i * width / len(data),
                  data.loc[name, ['demDP_school', 'disIP_school', 'eqOPP_school']],
                  width / len(data),
                  color=colors[i],
                  label=name)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=10)
    ax[1].set_title('Fairness Metrics across High School Type', size=14)

    for i, name in enumerate(list(data.index)):
        ax[2].bar(x + i * width / len(data), data.loc[name, ['demDP_elite', 'disIP_elite', 'eqOPP_elite']], width / len(data), color=colors[i])

    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, fontsize=10)
    ax[2].set_title('Fairness Metrics across Gender-High School Type', size=14)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                        bottom=0.1)
    ax.flatten()[1].legend(loc='upper center', frameon=False,
                           bbox_to_anchor=(0.0, -0.25), ncol=len(data), prop={'size': 12})

    fig.tight_layout(pad=2.0)

    plt.savefig(f'{outfolder}/baseline_metrics.png')

if __name__ == '__main__':
    datafile = '/home/mx/Documents/Xavier/EDM/FairEd/results/exploration/baseline_stress_test.csv'
    outfolder = '../results/stress_test'

    os.makedirs(outfolder, exist_ok=True)

    plot_fairness_metric_baseline(datafile, outfolder)
