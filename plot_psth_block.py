#psth tracking vs playback par cluster et save le plot

from kneed import DataGenerator, KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import scipy.io
import json
import numpy as np
from format_data import *
from utils import *
import pickle
from matplotlib.colors import LinearSegmentedColormap

t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
#gc = np.arange(0, 32)

path = '/auto/data2/eTheremin/MUROLS/MUROLS_20230220/MUROLS_20230220_SESSION_00/'

data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
#gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)
gc = np.arange(32)

n_block = int(np.max([elt['Block'] for elt in features]))

from matplotlib.colors import LinearSegmentedColormap

# Créer une colormap allant du rouge foncé au rouge clair
colors = [(0.5, 0, 0), (1, 0.6, 0.6)]  # rouge foncé -> rouge clair
cmap = LinearSegmentedColormap.from_list('red_scale', colors, N=5)

for i in range(1, 6):
    tracking = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'tracking')
    #playback = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'playback')
    c_tracking = np.nanmean(tracking, axis=1)
    #c_playback = np.nanmean(playback, axis=1)
    m_tracking = np.nanmean(c_tracking, axis=0)
    #m_playback = np.nanmean(c_playback, axis=0)
    
    # Appliquer la couleur de l'échelle en fonction du bloc
    plt.plot(m_tracking, label=f'block {i}', color=cmap(i / 5))
plt.xlabel('Time [s]')
plt.ylabel('[spikes/s]')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.title('Evolution of tracking block by block')

plt.savefig(path+'headstage_0/tracking_evolution.png')
plt.close()


# Créer une colormap allant du noir au gris clair
colors = [(0, 0, 0), (0.8, 0.8, 0.8)]  # noir -> gris clair
cmap = LinearSegmentedColormap.from_list('gray_scale', colors, N=5)

for i in range(1, n_block+1):
    #tracking = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'tracking')
    playback = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, i, 'playback')
    #c_tracking = np.nanmean(tracking, axis=1)
    c_playback = np.nanmean(playback, axis=1)
    #m_tracking = np.nanmean(c_tracking, axis=0)
    m_playback = np.nanmean(c_playback, axis=0)
    
    # Appliquer la couleur de l'échelle en fonction du bloc
    plt.plot(m_playback, label=f'block {i}', color=cmap(i / 5))

plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('[spikes/s]')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Evolution of playback block by block')

plt.savefig(path+'headstage_0/playback_evolution.png')
plt.close()


for block in range(1, n_block+1):
    tracking = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, block, 'tracking')
    playback = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, block, 'playback')
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(f'tracking vs playback for block {block}', y=1.02)
    plt.subplots_adjust() 
    num_plots, num_rows, num_columns = get_better_plot_geometry(gc)
    psth_bins = np.arange(-t_pre, t_post, bin_width)
    for n, cluster in enumerate(num_plots):
        if cluster < num_plots: 
            row, col = get_plot_coords(cluster)
            axes[row, col].plot(psth_bins, np.nanmean(tracking[n], axis=0), c = 'red')
            axes[row, col].plot(psth_bins, np.nanmean(playback[n], axis=0), c = 'black')
            axes[row, col].axvline(0, c = 'grey', linestyle='--')
            axes[row, col].set_title(f'Cluster {cluster}')
            axes[row, col].spines['top'].set_visible(False)
            axes[row, col].spines['right'].set_visible(False)
    plt.title(f'tracking vs playback for block {block}')
    plt.savefig(path+f'headstage_0/psth_cluster_block_{block}.png')
    plt.close()