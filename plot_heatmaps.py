from utils import *
from tonotopy import *
import findpeaks
from skimage import measure
import os

path = "/Volumes/data2/eTheremin/ALTAI/ALTAI_20240806_SESSION_00/"

#session = 'MMELOIK_20241029_SESSION_00'
#path = '/Volumes/data6/eTheremin/MMELOIK/'+ session + '/'


t_pre = 0.5#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
condition = 'tracking' #or playback

data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
#gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)
gc = np.arange(32)

tones = get_played_frequency(features, t_pre, t_post, bin_width, condition)
# prendre les valeurs uniques de tones
unique_tones = sorted(np.unique(tones))

# ne calculer les heatmaps uniquement si on ne trouve pas le fichier heatmaps.npy
if not os.path.exists(path + 'heatmap_plot_playback.npy'):
    print("calculating heatmaps")
    heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, 0, 0, condition, 'heatmaps')

else:
    heatmaps = np.load(path + 'heatmap_plot_playback.npy', allow_pickle = True)
    print('heatmaps already exist')

#récupérer les heatmaps
plot_heatmap_bandwidth(heatmaps,3, gc,unique_tones, 2, 2, bin_width, psth_bins, t_pre,path, '', condition)