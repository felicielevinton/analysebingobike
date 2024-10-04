import numpy as np
from scipy.io import savemat
import scipy.io as sio
import os


#path = 'Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/' 
# path = 'Z:/eTheremin/ALTAI/ALTAI_20240722_SESSION_01/'
path = 'Z:/eTheremin/ALTAI/ALTAI_20240710_SESSION_00/'
session = 'ALTAI_20240710_SESSION_00'
save_path = 'Y:/eTheremin/clara/ALTAI_20240710_SESSION_00/' +'std.min = 4 bis/'
#path = 'Z:/eTheremin/ALTAI/ALTAI_20240914_SESSION_00/'
data_t = np.load(path + 'headstage_0/filtered_neural_data.npy', allow_pickle = True)
data = data_t.transpose() # pour les données filrées
print(data.shape)


# try:
#     os.makedirs(save_path + session, exist_ok=True)
#     print(f"Dossiers créés avec succès : {save_path + session}")
# except OSError as e:
#     print(f"Erreur lors de la création du dossier : {e}")

#num_channel = [13,0,4,15,10,19,9,22,12,7,11]
#num_channel = [3,6,17,5,23,16,14,31]
num_channel = np.load(path + 'headstage_0/good_clusters.npy', allow_pickle = True)
print(num_channel)
for k in num_channel:
    print(k)
    data_C = data[k,:]
    data_dict = {'data': data_C,'sr':30000}
    savemat(save_path + 'C'+ str(k) +'.mat',data_dict)
    print('ok')







# # Créer le dossier spike_sorting

# try:
#     os.makedirs(path + 'spike_sorting', exist_ok=True)
#     os.makedirs(path + 'headstage_0/spike_sorting', exist_ok=True)
#     print(f"Dossiers créés avec succès : {path}")
# except OSError as e:
#     print(f"Erreur lors de la création du dossier : {e}")

