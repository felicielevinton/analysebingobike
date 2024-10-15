import numpy as np
from scipy.signal import find_peaks
import os
import json


def est_premier(nombre):
    if nombre <= 1:
        return False
    elif nombre <= 3:
        return True
    elif nombre % 2 == 0 or nombre % 3 == 0:
        return False
    i = 5
    while i * i <= nombre:
        if nombre % i == 0 or nombre % (i + 2) == 0:
            return False
        i += 6
    return True

def get_plot_coords(channel_number):
    """
    Fonction qui calcule la position en 2D d'un canal sur une Microprobe.
    Retourne la ligne et la colonne.
    """
    if channel_number in list(range(8)):
        row = 3
        col = channel_number % 8

    elif channel_number in list(range(8, 16)):
        row = 1
        col = 7 - channel_number % 8

    elif channel_number in list(range(16, 24)):
        row = 0
        col = 7 - channel_number % 8

    else:
        row = 2
        col = channel_number % 8

    return row, col



def get_plot_geometry(good_clusters):
    n_clus = len(good_clusters)
    if est_premier(n_clus):
        n_clus=n_clus-1

    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
        
        #print(num_columns)
    num_rows = -(-n_clus // num_columns)
    return num_rows, num_columns

def get_better_plot_geometry(good_clusters):
    # Calculate number of rows and columns for subplots
    num_plots = len(good_clusters)
    num_cols = int(np.ceil(np.sqrt(num_plots)))
    num_rows = int(np.ceil(num_plots / num_cols))
    return num_plots, num_rows, num_cols

def get_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth


def get_psth_in_block(data, features, t_pre, t_post, bin_width, good_clusters, block, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Block']==block:
                    if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth

def get_played_frequency(features, t_pre, t_post, bin_width, condition):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    if condition=="tracking":
        c = 0
    elif condition=="playback":
        c=1
    elif condition=="tail":
        c = -1
    elif condition == "mappingchange":
        c = 2
    frequency = []
    for bin in range(len(features)):
        if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                frequency.append(features[bin]['Played_frequency'])
    return frequency
        


def get_mock_frequency(features):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    c=1
    frequency = []
    for bin in range(len(features)):
        if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
            frequency.append(features[bin]['Mock_frequency'])
    return frequency
        






def get_sustained_activity(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    return (np.nanmean(psth[0: int(t_pre/bin_width)-2]))




def get_sustained_activity_nan(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    --> dans la cas où on aurait des nan gênants
    
    
    """
    if psth is not np.nan:
    
        return (np.nanmean(psth[0: int(t_pre/bin_width)-2]))
    else:
        return np.nan




def mean_maxima(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau dont les indices sont compris
    entre t0 et t1
    """
    # Find peaks in the array
    pics, _ = find_peaks(arr[t0:t1], distance=thresh)

    # Check if there are at least two peaks
    if len(pics) >= 2:
        # Get the indices of the two maximum values
        max_indices = np.argsort(arr[pics])[-2:]
        # Calculate the mean of the two maximum values
        #mean = np.mean(arr[pics][max_indices])
        mean = np.max(arr[pics][max_indices])

        # Get the actual maximum values
        max_values = arr[pics][max_indices]
    else:
        mean = np.nan
        max_values = np.nan

    return mean, pics, max_values


def mean_maxima_nan(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau cont les indices sont compris
    entre t0 et t1
    
    --> cas où on aurait des nan gênants
    """
    # Find peaks in the array
    if arr is not np.nan:
        pics, _ = find_peaks(arr, distance=thresh)

        # Check if there are at least two peaks
        if len(pics) >= 2:
            # Get the indices of the two maximum values
            max_indices = np.argsort(arr[pics])[-2:]

            # Calculate the mean of the two maximum values
            #mean = np.mean(arr[pics][max_indices])
            mean = np.max(arr[pics][max_indices])
            # Get the actual maximum values
            max_values = arr[pics][max_indices]
        else:
            mean = np.nan
            max_values = np.nan
    else:
        mean = np.nan
        max_values = np.nan
        pics=np.nan
        

    return mean, pics, max_values


def get_total_evoked_response(psth, t_pre, t_post, bin_width, thresh, t0, t1):
    """"
    Function qui renvoie la total evoked reponse pour un tableau contenant des psth
    input : un tableau psth contenant des psth
    output : un tableau contenant la total evoked response pour chaque psth
    
    """
    total_evoked_response = []
    for elt in psth:
        total_evoked_response.append(mean_maxima(elt, thresh, t0,t1)[0])
        #total_evoked_response.append(np.max(elt))
    return total_evoked_response


def get_indexes(tableau, a):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est égale à a

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i] == a:
            indices_a.append(i)

    return indices_a

def get_indexes_in(tableau, a, b):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est comprise entre a et b

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i]>=a and tableau[i]<=b:
            indices_a.append(i)

    return indices_a


def get_psth_for_indexes(data, features, indexes, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), indexes (les indices des bin qui nous intéressent)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in indexes:
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth



def get_mean_psth_in_bandwidth(data, features, bandwidth, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, renvoie la moyenne des psth pour toutes les fréquences comprises dans la badnwidth du cluster
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), bandwidth
    output : 
     - une liste contenant le psth moyen par cluster [cluster x [t_pre, t_post] ] in la bandwidth
      et une autre out la bandwidth
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    else : 
        c=1
        
    
    in_psth, out_psth=[] , []
    for idx, cluster in enumerate(good_clusters):
        psth_clus, out_clus = [], []
        low_f, high_f = bandwidth[idx][0],  bandwidth[idx][1]
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c:
                    if low_f<=features[bin]['Played_frequency']<=high_f:
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    else:
                        out_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        if len(psth_clus)==0:
            psth_clus = [[np.nan]*(len(psth_bins)-1)]*2
        if len(out_clus)==0:
            out_clus = [[np.nan]*(len(psth_bins)-1)]*2
        in_psth.append(np.nanmean(psth_clus, axis=0))
        out_psth.append(np.nanmean(out_clus, axis=0))
       
    return in_psth, out_psth    


def get_sem(neurones):
    """""
    Fonction qui renvoie la sem pour un tableau de format (neurones x bin)
    
    input : un tableau [neurones, bins]
    output: liste [bins] contenant la SEM
    """
    sem = []
    for bin in range(len(neurones[0])):
        sem.append(np.nanstd(np.array(neurones)[:,bin])/np.sqrt(len(neurones)))
    return sem  










def get_sustained_activity_OLD(psth, t_pre, t_post, bin_width):
    """""
    PAS UTILE POUR L'INSTANT !!!
    Fonction qui renvoie l'activité moyenne d'un tableau de PSTH
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    sustained = []
    for elt in psth:
        sustained.append(np.nanmean(elt[0: int(t_pre/bin_width)-2]))
    return sustained 


def indices_valeurs_egales(tableau, valeur_cible):
    """
    

    Args:
        tableau (_type_): un tableau
        valeur_cible (_type_): la valeur qu'on recherche dans le tableau

    Returns:
        indices: les indices des éléments dans le tableau dont la valeur est égale à la valeur cible
    """
    indices = []
    for i in range(len(tableau)):
        if tableau[i] == valeur_cible:
            indices.append(i)
    return indices


def indices_valeurs_comprises(tableau, valeur_min, valeur_max):
    
    """"
       Args:
        tableau (_type_): un tableau
        valeur_min, valeur_max (_type_): valeurs qui définissent l'intervalle dans lequel on cherche des valeurs dans le tableau

        Returns:
            indices: les indices des éléments dans le tableau dont la valeur est comprise dans l'intervalle.
    """
    indices = []
    for i in range(len(tableau)):
        if valeur_min<=tableau[i]<valeur_max:
            indices.append(i)
    return indices




def get_mean_psth_in_bandwidth(data, features, bandwidth, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, renvoie la moyenne des psth pour toutes les fréquences comprises dans la badnwidth du cluster
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), bandwidth
    output : 
     - une liste contenant le psth moyen par cluster [cluster x [t_pre, t_post] ] in la bandwidth
      et une autre out la bandwidth
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
        
    
    in_psth, out_psth=[] , []
    for idx, cluster in enumerate(good_clusters):
        psth_clus, out_clus = [], []
        low_f, high_f = bandwidth[idx][0],  bandwidth[idx][1]
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c:
                    if low_f<=features[bin]['Played_frequency']<=high_f:
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    else:
                        out_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        if len(psth_clus)==0:
            psth_clus = [[np.nan]*(len(psth_bins)-1)]*2
        if len(out_clus)==0:
            out_clus = [[np.nan]*(len(psth_bins)-1)]*2
        in_psth.append(np.nanmean(psth_clus, axis=0))
        out_psth.append(np.nanmean(out_clus, axis=0))
       
    return in_psth, out_psth    
 
    

def get_session_type_final(path):
    """
    Fonction qui renvoie le type de la session parmi TrackingOnly, PlaybackOnly etc
    elle va chercher dans le fichier json le type de session
    """
    # List all files in the folder
    files = os.listdir(path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]
    # Check if only one JSON file is found
    if len(json_files) == 1:
        json_file_path = os.path.join(path, json_files[0])
        print("Found JSON file:", json_file_path)
        # Load the JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        # Extract the "Type" field
        try : 
            type_value = data['Block_000']['Type']

            if type_value=="Pause":
                type_value = data['Block_001']['Type']
                
            print("Type:", type_value)
        except :
            type_value = [data[key]["Type"] for key in data if key.startswith("Experiment_")][1]
            print("Type:", type_value)       
            
            
    else:
        print("Error: No JSON files found.")
    return type_value



def get_mean_neurone_spaced_frequency(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par neurone
    Attention ici je ne prends que les changements de fréquence qui sont 
    séparés de plus de 200ms (pour vérifier que les oscillations sont bien
    dûes aux changements de fréquence précédents le stim d'intéret)
    --> si tu veux l'utiliser : change l'appel à la fonction dans get_mean_psth
    input: fichier data.npy d'une session, features.npy, t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        previousbin=0
        for bin in range(len(features)):
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0 and bin-previousbin>0.2/bin_width:
                    mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    previousbin=bin
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and bin-previousbin>0.2/bin_width:
                    mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    previousbin=bin
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback