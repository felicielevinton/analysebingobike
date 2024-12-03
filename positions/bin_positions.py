from kneed import DataGenerator, KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import scipy.io
from scipy.signal import find_peaks
import json
import numpy as np
import pickle
import resync as rs
from mapping import *
from testing import *
import json
from scipy.stats import mode

bin_width = 0.005
fs = 160



def clean_positions(positions):
    """
    Fonction de nettoyage des positions enregistrées au cours de l'expérience. Pour rappel,
    une valeur de -1 indique que le sujet n'a pas été détecté par le réseau de neurones.
    Cette version remplace toutes les occurrences de -1, même celles qui sont à la fin ou dans une séquence continue.
    
    :param positions: Tableau des positions avec potentiellement des valeurs -1
    :return: Tableau des positions avec toutes les valeurs -1 remplacées
    """
    
    # Identifier les indices où positions == -1
    y = np.where(positions == -1)[0]
    
    if len(y) == 0:
        # Aucun -1 à remplacer
        return positions
    
    # Parcourir toutes les séquences de -1 et les remplacer
    k = 0
    begin = None
    for i in range(len(y)):
        if i == 0 or y[i] == y[i - 1] + 1:
            # Début ou continuité d'une séquence de -1
            if k == 0:
                begin = y[i]  # Enregistrer le début de la séquence
            k += 1
        else:
            # Fin d'une séquence de -1
            end = y[i - 1]
            filler_value = positions[begin - 1] if begin > 0 else positions[end + 1]
            positions[begin:end + 1] = filler_value
            k = 1
            begin = y[i]
    
    # Traiter la dernière séquence de -1, s'il y en a
    if k > 0:
        end = y[-1]
        filler_value = positions[begin - 1] if begin > 0 else positions[end + 1]
        positions[begin:end + 1] = filler_value

    return positions


session = 'MMELOIK_20241128_SESSION_00'
path = '/Volumes/data6/eTheremin/MMELOIK/'+ session + '/positions'

folder = '/Volumes/data6/eTheremin/MMELOIK/'+ session +'/'

features = np.load(folder+'headstage_0/features_0.005.npy', allow_pickle=True)
unique_tones = np.load(folder+'headstage_0/unique_tones.npy', allow_pickle=True)


# Lire le json pour trouver les fichiers de positions

# trouver le chemin vers le json
json_path = folder+'/session_MMELOIK_SESSION_00_20241128.json'

# Dictionnaires pour stocker les valeurs de 'Positions_fn' et 'playback' Positions_fn
positions_tracking = {}
positions_playback = {}

# Charger le fichier JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Dictionnaire pour stocker les listes concaténées de fichiers
positions_concatenated = {}

# Parcourir chaque bloc et construire les listes concaténées
for block, content in data.items():
    if block.startswith("Block"):
        # Créer une liste vide pour ce bloc
        positions_concatenated[block] = []
        
        # Ajouter le fichier de tracking s'il existe
        if "Positions_fn" in content:
            positions_concatenated[block].append(content["Positions_fn"])
        elif "tracking" in content and "Positions_fn" in content["tracking"]:
            positions_concatenated[block].append(content["tracking"]["Positions_fn"])
        
        # Ajouter le fichier de playback s'il existe
        if "playback" in content and "Positions_fn" in content["playback"]:
            positions_concatenated[block].append(content["playback"]["Positions_fn"])

# stocker les positions  et leur condition associée dans des tableaux : 

positions_total, tracking_positions, mock_positions = [], [], []
condition_total, block_total = [],[]
len_tr, len_pb = [],[]
for block in positions_concatenated:
    i = int(block.split('_')[-1]) 
    print(block)
    position_files = positions_concatenated[block]
    print(position_files)
    if len(position_files)>1:
        block_tr = np.fromfile(path+'/'+ position_files[0], dtype=np.int32)
        block_pb = np.fromfile(path+'/'+ position_files[1], dtype=np.int32)
        x_tr = block_tr[np.arange(0, len(block_tr), step=2)]
        x_pb = block_pb[np.arange(0, len(block_pb), step=2)]
        tracking_positions.append(clean_positions(x_tr))
        #tracking_block.append(np.zeros(len(rs.clean_positions(x_tr))))
        mock_positions.append(clean_positions(x_pb))
        #mock_block.append(np.ones(len(rs.clean_positions(x_pb))))
        positions_total.append(clean_positions(x_tr))
        positions_total.append(clean_positions(x_pb))
        condition_total.append(np.zeros(len(clean_positions(x_tr))))
        condition_total.append(np.ones(len(clean_positions(x_pb))))
        length_block = len(clean_positions(x_tr)) + len(clean_positions(x_pb)) 
        block_total.append(np.full(length_block, i))

    else : 
        block_tail = np.fromfile(path+'/'+ position_files[0], dtype=np.int32)
        x_tail = block_tail[np.arange(0, len(block_tail), step=2)]
        #tracking_block.append(np.full_like(x_tail, -1))
        positions_total.append(clean_positions(x_tail))
        tracking_positions.append(clean_positions(x_tail))
        condition_total.append(np.full_like(x_tail, -1))
        block_total.append(np.full(len(clean_positions(x_tail)), i))

# positions_total contient block par block les positions nettoyées, block_total le block associé et condition_total, la condition.
positions_total = np.hstack(positions_total)
block_total = np.hstack(block_total)
condition_total = np.hstack(condition_total)

# on binne les positions : 
mapping = Mapping(1920, 33, 2000., 7)

# ici je binne les positions
t_original = np.arange(len(positions_total)) / fs
t_bins = np.arange(0, t_original[-1], bin_width)
binned_positions = np.interp(t_bins, t_original, positions_total)

# convertir les positions en fréquences
binned_tones = mapping.convert_to_frequency(np.array(binned_positions, dtype=int))

# Binner les blocks et condition liés aux positions

# Calculer l'indice du bin correspondant à chaque temps original
bin_indices = np.digitize(t_original, t_bins) - 1  # -1 car digitize renvoie des indices à partir de 1

# Initialiser un tableau pour les blocs binnés
binned_blocks = np.zeros(len(t_bins), dtype=int)

for i in range(len(t_bins)):
    # Sélectionner les indices appartenant au bin actuel
    block_indices_in_bin = (bin_indices == i)
    if np.any(block_indices_in_bin):  # Vérifier si le bin contient des éléments
        # Calculer le mode en s'assurant que le résultat est valide
        current_mode = mode(block_total[block_indices_in_bin], keepdims=True)
        binned_blocks[i] = current_mode.mode[0] if current_mode.count[0] > 0 else -1
    else:
        binned_blocks[i] = binned_blocks[i-1] # Valeur par défaut si le bin est vide (par exemple -1)

binned_condition = np.zeros(len(t_bins), dtype=int)
for i in range(len(t_bins)):
    # Sélectionner les indices appartenant au bin actuel
    condition_indices_in_bin = (bin_indices == i)
    if np.any(condition_indices_in_bin):  # Vérifier si le bin contient des éléments
        # Calculer le mode en s'assurant que le résultat est valide
        current_mode = mode(condition_total[condition_indices_in_bin], keepdims=True)
        binned_condition[i] = current_mode.mode[0] if current_mode.count[0] > 0 else -1
    else:
        binned_condition[i] = binned_condition[i-1] # Valeur par défaut si le bin est vide (par exemple -1)
    

print(len(binned_condition))
print(len(binned_blocks))
print(len(binned_positions))
print(len(features))


target_length = len(features)

# Fonction pour compléter un tableau avec des np.nan
def pad_with_nan(array, target_length):
    if len(array) < target_length:
        # Calcul de combien de np.nan sont nécessaires
        padding = target_length - len(array)
        # Ajouter des np.nan à la fin
        return np.concatenate([array, np.full(padding, np.nan)])
    return array  # Si la longueur est déjà correcte, on ne fait rien

# Appliquer la fonction sur chaque tableau
binned_condition = pad_with_nan(binned_condition, target_length)
binned_blocks = pad_with_nan(binned_blocks, target_length)
binned_positions = pad_with_nan(binned_positions, target_length)
binned_tones = pad_with_nan(binned_tones, target_length)

# Les tableaux sont maintenant de la même longueur que features
print(len(binned_condition))
print(len(binned_blocks))
print(len(binned_positions))
print(len(features))


for i, feature in enumerate(features[:len(features)]):
    feature['Position'] = binned_positions[i]
    feature['Position_frequency'] = binned_tones[i]
    feature['Position_condition'] = binned_condition[i]
    feature['Position_block'] = binned_blocks[i]


#features = list(features.values())
np.save(folder+f'/features_wmotion_{bin_width}.npy', features)


