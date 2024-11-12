from utils import *
from tonotopy import *
import findpeaks
from skimage import measure
import os

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
path = "/Volumes/data2/eTheremin/ALTAI/ALTAI_20240809_SESSION_00/"

#session = 'MMELOIK_20241029_SESSION_00'
#path = '/Volumes/data6/eTheremin/MMELOIK/'+ session + '/'

#session = 'MUROLS_20230227/MUROLS_20230227_SESSION_00'
#path = '/Volumes/data2/eTheremin/MUROLS/'+ session + '/'
=======
#path = "/auto/data6/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240709_SESSION_01/"
>>>>>>> e15e68c1 (autre)

session = '/MUROLS_20230220/MUROLS_20230220_SESSION_00'
path = '/auto/data2/eTheremin/MUROLS/'+ session + '/'
=======
#path = "/auto/data6/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240709_SESSION_01/"
#session = '/MMELOIK_20241107_SESSION_00'
# path = '/auto/data6/eTheremin/MMELOIK/'+ session + '/'

path = '/auto/data2/eTheremin/MUROLS/MUROLS_20230221/MUROLS_20230221_SESSION_00'
>>>>>>> 5fb8e761 (nouvelle méthode pour in/out qui marche bien)
=======
#path = "/auto/data6/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240709_SESSION_01/"
#session = '/MMELOIK_20241107_SESSION_00'
# path = '/auto/data6/eTheremin/MMELOIK/'+ session + '/'

path = '/auto/data2/eTheremin/ALTAI/ALTAI_20240806_SESSION_00/'
>>>>>>> 283d6daf (pôsitions)

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
tones = [int(x) for x in tones]
# prendre les valeurs uniques de tones
unique_tones = sorted(np.unique(tones))
unique_tones = [int(x) for x in unique_tones]

# ne calculer les heatmaps uniquement si on ne trouve pas le fichier heatmaps.npy
if not os.path.exists(path + f'heatmap_plot_{condition}.npy'):
    print("calculating heatmaps")
    heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, 0, 0, condition, 'heatmaps')

else:
    heatmaps = np.load(path + f'heatmap_plot_{condition}.npy', allow_pickle = True)
    print('heatmaps already exist')

#récupérer les heatmaps
plot_heatmap_bandwidth(heatmaps,3, gc,unique_tones, 2, 2, bin_width, psth_bins, t_pre,path, '', condition)