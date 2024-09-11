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

t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
#gc = np.arange(0, 32)

path = '/mnt/working2/felicie/data2/eTheremin/ALTAI/ALTAI_20240814_SESSION_00/'

data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
#gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)
gc = np.arange(0, 32)

tail = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tail')
tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
mc = get_psth(data, features, t_pre, t_post, bin_width, gc, 'mapping change')
playback = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback') 

np.save(path+'headstage_0/psth_tracking_0.005.npy', tracking)
np.save(path+'headstage_0/psth_playback_0.005.npy', playback)
np.save(path+'headstage_0/psth_mappingchange_0.005.npy', mc) 
np.save(path+'headstage_0/psth_tail_0.005.npy', tail)


# pour plot cluster par cluster16
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('tracking vs playback', y=1.02)
plt.subplots_adjust() 
num_plots, num_rows, num_columns = get_better_plot_geometry(gc)
psth_bins = np.arange(-t_pre, t_post, bin_width)
for cluster in range(num_plots):
    if cluster < num_plots: 
        row, col = get_plot_coords(cluster)
        axes[row, col].plot(psth_bins, np.nanmean(tracking[cluster], axis=0), c = 'red')
        axes[row, col].plot(psth_bins, np.nanmean(playback[cluster], axis=0), c = 'black')
        axes[row, col].axvline(0, c = 'grey', linestyle='--')
        axes[row, col].set_title(f'Cluster {cluster}')
plt.savefig(path+'headstage_0/psth_cluster.png')
plt.close()

# la moyenne sur tous les clusters
c_tracking = np.nanmean(tracking, axis=0)
c_playback = np.nanmean(playback, axis=0)

m_tracking = np.nanmean(c_tracking, axis=0)
m_playback = np.nanmean(c_playback, axis=0)

sem_tr = get_sem(c_tracking)
sem_pb = get_sem(c_playback)

plt.plot(psth_bins, m_tracking, c = 'red', label = 'tracking')
plt.plot(psth_bins, m_playback, c = 'black',  label = 'playback')
plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='red', alpha=0.2)
plt.fill_between(psth_bins, m_playback - sem_pb, m_playback + sem_pb, color='black', alpha=0.2)
plt.title('Tracking vs playback (Average over all clusters)')
plt.xlabel('Time [s]')
plt.ylabel('[spikes/s]')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()

plt.savefig(path+'headstage_0/psth_average.png')