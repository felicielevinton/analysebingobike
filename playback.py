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
import math


t_pre = 0.2#0.2
t_post = 0.30#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)

path = '/auto/data2/eTheremin/ALTAI/ALTAI_20240822/ALTAI_20240822_SESSION_00/'
bb = True

data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)

n_block = int(np.max([elt['Block'] for elt in features]))

cols = 3  # Fixed number of columns (for example)
rows = math.ceil(n_block / cols)

# END tr vs END pb
fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
fig.suptitle('End of tracking vs end of playback', y=1.02)
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

# Flatten axes array to simplify indexing (in case of 1D or 2D axes)
axes = axes.flatten()

if bb == True:   #juste pour pas tout plotter a chaque fois 

    for bloc in range(1,n_block):  # Loop over the actual number of blocs
        ax = axes[bloc-1]  # Access the subplot for the current bloc
        print(bloc)
        tr = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking')
        pb = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback')

        tr = np.array(tr)
        pb = np.array(pb)
        
        n = int(len(tr[0]) / 5)
        print(n)
        
        end_tr = tr[:, 4 * n:]
        end_pb = pb[:, 4 * n:]

        mc_tr = np.nanmean(end_tr, axis=1)
        mc_pb = np.nanmean(end_pb, axis=1)

        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)

        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)

        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[spikes/s]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(path+'headstage_0/end_end.png', bbox_inches='tight')
    plt.close()
    print('ok end to end saved')

    # BEGINNING vs BEGINNING

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle('Beginning of tracking vs beginning of playback', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

    # Flatten axes array to simplify indexing (in case of 1D or 2D axes)
    axes = axes.flatten()

    for bloc in range(1,n_block):  # Loop over the actual number of blocs
        ax = axes[bloc-1]  # Access the subplot for the current bloc
        print(bloc)
        tr = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking')
        pb = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback')

        tr = np.array(tr)
        pb = np.array(pb)
        
        n = int(len(tr[0]) / 5)
        print(n)
        
        beg_tr = tr[:, :n]
        beg_pb = pb[:, :n]

        mc_tr = np.nanmean(beg_tr, axis=1)
        mc_pb = np.nanmean(beg_pb, axis=1)

        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)

        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)

        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[spikes/s]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block+1:]:
        ax.axis('off')
    plt.savefig(path+'headstage_0/beg_beg.png', bbox_inches='tight')
    plt.close()

    print('ok beg to beg saved')

    # END tracking vs BEGINNING playback

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    fig.suptitle('End of tracking vs beginning of playback', y=1.02)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

    # Flatten axes array to simplify indexing (in case of 1D or 2D axes)
    axes = axes.flatten()

    for bloc in range(1,n_block):  # Loop over the actual number of blocs
        ax = axes[bloc-1]  # Access the subplot for the current bloc
        print(bloc)
        tr = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking')
        pb = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback')

        tr = np.array(tr)
        pb = np.array(pb)
        
        n = int(len(tr[0]) / 5)
        print(n)
        
        end_tr = tr[:,4 * n:]
        beg_pb = pb[:, :n]

        mc_tr = np.nanmean(end_tr, axis=1)
        mc_pb = np.nanmean(beg_pb, axis=1)

        m_tr = np.nanmean(mc_tr, axis=0)
        m_pb = np.nanmean(mc_pb, axis=0)

        sem_tr = get_sem(mc_tr)
        sem_pb = get_sem(mc_pb)

        ax.plot(psth_bins, m_tr, c='red', label='Tracking')
        ax.plot(psth_bins, m_pb, c='black', label='Playback')
        
        ax.fill_between(psth_bins, m_tr - sem_tr, m_tr + sem_tr, color='red', alpha=0.2)
        ax.fill_between(psth_bins, m_pb - sem_pb, m_pb + sem_pb, color='black', alpha=0.2)

        ax.set_title(f'Block {bloc}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[spikes/s]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()

    for ax in axes[n_block:]:
        ax.axis('off')
    plt.savefig(path+'headstage_0/end_beg.png', bbox_inches='tight')
    plt.close()


# diviser le tracking en 2 et le playback en 2 

fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
fig.suptitle('Evolution of tracking in a block', y=1.02)
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

# Flatten axes array to simplify indexing (in case of 1D or 2D axes)
axes = axes.flatten()

for bloc in range(1,n_block):  # Loop over the actual number of blocs
    ax = axes[bloc-1]  # Access the subplot for the current bloc
    print(bloc)
    tr = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'tracking')
    #pb = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback')

    tr = np.array(tr)
    #pb = np.array(pb)
    
    n = int(len(tr[0]) / 2)
    print(n)
    
    beg_tr = tr[:,:n]
    end_tr = tr[:, n:]

    mc_beg_tr = np.nanmean(beg_tr, axis=1)
    mc_end_tr = np.nanmean(end_tr, axis=1)

    m_beg_tr = np.nanmean(mc_beg_tr, axis=0)
    m_end_tr = np.nanmean(mc_end_tr, axis=0)

    sem_beg_tr = get_sem(mc_beg_tr)
    sem_end_tr = get_sem(mc_end_tr)

    ax.plot(psth_bins, m_beg_tr, c='orange', label='First half tracking')
    ax.plot(psth_bins, m_end_tr, c='red', label='Second half tracking')
    
    ax.fill_between(psth_bins, m_beg_tr - sem_beg_tr, m_beg_tr + sem_beg_tr, color='orange', alpha=0.2)
    ax.fill_between(psth_bins, m_end_tr - sem_end_tr, m_end_tr + sem_end_tr, color='red', alpha=0.2)

    ax.set_title(f'Block {bloc}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[spikes/s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

for ax in axes[n_block:]:
    ax.axis('off')
plt.savefig(path+'headstage_0/tracking_didvided.png', bbox_inches='tight')
plt.close()
print("tracking saved")


# diviser le playback en 2 

fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
fig.suptitle('Evolution of playback in a block', y=1.02)
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between subplots

# Flatten axes array to simplify indexing (in case of 1D or 2D axes)
axes = axes.flatten()

for bloc in range(1,n_block):  # Loop over the actual number of blocs
    ax = axes[bloc-1]  # Access the subplot for the current bloc
    print(bloc)
    pb = get_psth_in_block(data, features, t_pre, t_post, bin_width, gc, bloc, 'playback')

    pb = np.array(pb)
    
    n = int(len(pb[0]) / 2)
    print(n)
    
    beg_pb = pb[:,:n]
    end_pb = pb[:, n:]

    mc_beg_pb = np.nanmean(beg_pb, axis=1)
    mc_end_pb = np.nanmean(end_pb, axis=1)

    m_beg_pb = np.nanmean(mc_beg_pb, axis=0)
    m_end_pb = np.nanmean(mc_end_pb, axis=0)

    sem_beg_pb = get_sem(mc_beg_pb)
    sem_end_pb = get_sem(mc_end_pb)

    ax.plot(psth_bins, m_beg_pb, c='grey', label='First half playback')
    ax.plot(psth_bins, m_end_pb, c='black', label='Second half playback')
    
    ax.fill_between(psth_bins, m_beg_pb - sem_beg_pb, m_beg_pb + sem_beg_pb, color='grey', alpha=0.2)
    ax.fill_between(psth_bins, m_end_pb - sem_end_pb, m_end_pb + sem_end_pb, color='black', alpha=0.2)

    ax.set_title(f'Block {bloc}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[spikes/s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

for ax in axes[n_block:]:
    ax.axis('off')
plt.savefig(path+'headstage_0/playback_didvided.png', bbox_inches='tight')
plt.close()
print("playback saved")
