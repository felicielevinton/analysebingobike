from utils import *
import matplotlib.pyplot as plt

t_pre = 0.5#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)

# Pour créer des fichiers .npy qui contiennent les psth et ne pas avoir à les recalculer à chaque fois

playback_sessions = ['ALTAI_20240724_SESSION_01', 'ALTAI_20240724_SESSION_02', 'ALTAI_20240725_SESSION_00', 'ALTAI_20240726_SESSION_01',
                           'ALTAI_20240809_SESSION_00', 'ALTAI_20240814_SESSION_00', 'ALTAI_20240822_SESSION_00']

for session in playback_sessions:
    path = '/auto/data2/eTheremin/ALTAI/'+ session + '/'

    data = np.load(path+f'headstage_0/data_{bin_width}.npy', allow_pickle=True)
    features = np.load(path+f'headstage_0/features_{bin_width}.npy', allow_pickle=True)
    gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)
    tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
    playback = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback') 
    m_tracking = np.nanmean(tracking, axis=1)
    m_playback = np.nanmean(playback, axis=1)
    np.save(path+f'psth_tracking_{bin_width}.npy', m_tracking)
    np.save(path+f'psth_playback_{bin_width}.npy', m_playback)