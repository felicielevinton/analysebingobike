from utils import *
from tonotopy import *
import findpeaks
from skimage import measure
import os

path = "/auto/data6/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_01/"


t_pre = 0.5#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)

data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)

tones = get_played_frequency(features, t_pre, t_post, bin_width, 'playback')
# prendre les valeurs uniques de tones
unique_tones = sorted(np.unique(tones))

# ne calculer les heatmaps uniquement si on ne trouve pas le fichier heatmaps.npy
if not os.path.exists(path + 'heatmap_plot_playback.npy'):
    print("calculating heatmaps")
    heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, 0, 0, 'playback', 'heatmaps')

else:
    heatmaps = np.load(path + 'heatmap_plot_playback.npy', allow_pickle = True)
    print('heatmaps already exist')

#récupérer les heatmaps
plot_heatmap_bandwidth(heatmaps,3, gc,unique_tones, 2, 2, bin_width, psth_bins, t_pre,path, '', 'playback')