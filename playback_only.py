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

path = '/Volumes/data6/eTheremin/MMELOIK/MMELOIK_20241127_SESSION_01/headstage_1/'

data = np.load(path+'/data_0.005.npy', allow_pickle=True)
features = np.load(path+'/features_0.005.npy', allow_pickle=True)
gc = np.load(path+'/good_clusters.npy', allow_pickle=True)

tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback')

n = int(len(tracking[0])/2)
tracking = np.array(tracking)
first_tr = tracking[:,:n]
second_tr = tracking[:,n:]

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Evolution of playback', y=1.02)
plt.subplots_adjust() 
num_plots, num_rows, num_columns = get_better_plot_geometry(gc)
psth_bins = np.arange(-t_pre, t_post, bin_width)
for n, cluster in enumerate(gc):
    if n < num_plots: 
        row, col = get_plot_coords(cluster)
        axes[row, col].plot(psth_bins, np.nanmean(first_tr[n], axis=0), c = 'grey')
        axes[row, col].plot(psth_bins, np.nanmean(second_tr[n], axis=0), c = 'black')
        axes[row, col].axvline(0, c = 'grey', linestyle='--')
        axes[row, col].set_title(f'Cluster {cluster}')
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].spines['right'].set_visible(False)
#plt.legend()
plt.savefig(path+'/playback_evolution_psth_cluster.png')
plt.close()

# la moyenne sur tous les clusters
c_tracking = np.nanmean(first_tr, axis=0)
cc_tracking = np.nanmean(second_tr, axis=0)


m_tracking = np.nanmean(c_tracking, axis=0)
mm_tracking = np.nanmean(cc_tracking, axis=0)

sem_tr = get_sem(c_tracking)
sem_pb = get_sem(cc_tracking)

plt.plot(psth_bins, m_tracking, c = 'grey', label = 'first half playback')
plt.plot(psth_bins, mm_tracking, c = 'black',  label = 'second half playback')
plt.fill_between(psth_bins, m_tracking - sem_tr, m_tracking + sem_tr, color='grey', alpha=0.2)
plt.fill_between(psth_bins, mm_tracking - sem_pb, mm_tracking + sem_pb, color='black', alpha=0.2)
plt.title('Playback (Average over all clusters)')
plt.xlabel('Time [s]')
plt.ylabel('[spikes/s]')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend()
plt.savefig(path+'/playback_evolution_average.png')
plt.close()