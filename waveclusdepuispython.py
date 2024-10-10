import subprocess
import os
import time
import numpy as np

#good_channels = [3,24]



#sessions = ['ALTAI_20240711_SESSION_01','ALTAI_20240712_SESSION_01','ALTAI_20240722_SESSION_01','ALTAI_20240722_SESSION_04','ALTAI_20240724_SESSION_01',
 #           'ALTAI_20240724_SESSION_02']

sessions = ['ALTAI_20240710_SESSION_00']
for session in sessions:
    print(session)
    path = 'Y:/eTheremin/clara/' + session + '/' #+ 'filtered/std.min =5 bis/' 
    if os.path.exists('Z:/eTheremin/ALTAI/' + session + '/' + 'headstage_0/good_clusters.npy'):
        good_channels = np.load('Z:/eTheremin/ALTAI/' + session + '/' + 'headstage_0/good_clusters.npy', allow_pickle = True)
    else : 
        good_channels = np.arange(32)
    
    

    print(good_channels)

    # Intervalle de vérification (en secondes)+-
    interval_verification = 0.5
    timeout_creation = 300  # Temps maximum en secondes pour attendre la création des fichiers "times_C"

    for channel in good_channels:


        fichier_mat = path +  'C' + str(channel) + '.mat'
        fichier_spikes = path + 'C' + str(channel) + '_spikes.mat'
        fichier_times = path + 'times_C' + str(channel) + '.mat'  # Fichier créé par Do_clustering

        # Commandes pour MATLAB
        wave_clus_get_spikes = f"matlab -nodesktop -nosplash -batch \"cd('{path}'); Get_spikes('{fichier_mat}');\""
        wave_clus_do_clustering = f"matlab -nodesktop -nosplash -batch \"cd('{path}'); Do_clustering('{fichier_spikes}');\""

        # Exécuter la première commande wave_clus avec redirection des erreurs vers un fichier log
        try:
            print(f"Exécution de Get_spikes pour le canal {channel}...")
            with open('log_get_spikes.txt', 'a') as log_file:
                subprocess.run(wave_clus_get_spikes, shell=True, check=True, stdout=log_file, stderr=log_file)
            print(f"Get_spikes exécuté avec succès pour le canal {channel}.")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution de Get_spikes pour le canal {channel} : {e}")
            exit(1)

        # Attendre que le fichier de spikes soit généré avant de lancer Do_clustering
        while not os.path.exists(fichier_spikes):
            print(f"Attente de la création de {fichier_spikes} pour le canal {channel}...")
            time.sleep(interval_verification)

        # Exécuter la deuxième commande wave_clus avec redirection des erreurs vers un fichier log
        try:
            print(f"Exécution de Do_clustering pour le canal {channel}...")
            with open('log_do_clustering.txt', 'a') as log_file:
                subprocess.run(wave_clus_do_clustering, shell=True, check=True, stdout=log_file, stderr=log_file)
            print(f"Do_clustering exécuté avec succès pour le canal {channel}.")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution de Do_clustering pour le canal {channel} : {e}")
            exit(1)

        # Attendre que le fichier "times_C" soit créé
        time_start = time.time()
        while not os.path.exists(fichier_times):
            if time.time() - time_start > timeout_creation:
                print(f"Le fichier {fichier_times} n'a pas été créé pour le canal {channel} après {timeout_creation} secondes.")
                exit(1)
            print(f"Attente de la création de {fichier_times} pour le canal {channel}...")
            time.sleep(interval_verification)

        print(f"Le fichier {fichier_times} a été créé avec succès pour le canal {channel}.")




















# # good_channels_str = "[" + " ".join(map(str, good_channels)) + "]"
# # matlab_command = f"matlab -nodesktop -nosplash -r \"addpath('{script_path}');\"cd('{path}'); process_channels({good_channels_str}, '{path}'); exit;\""

# # # Exécuter la commande MATLAB une seule fois
# # try:
# #     print("Exécution du script MATLAB pour traiter tous les canaux...")
# #     with open('log_matlab_process.txt', 'w') as log_file:
# #         subprocess.run(matlab_command, shell=True, check=True, stdout=log_file, stderr=log_file)
# #     print("Traitement terminé avec succès.")
# # except subprocess.CalledProcessError as e:
# #     print(f"Erreur lors de l'exécution du script MATLAB : {e}")
