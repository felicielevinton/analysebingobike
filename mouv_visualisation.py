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

fs = 160

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

path = 'Z:/eTheremin/ALTAI/ALTAI_20240725_SESSION_00/'

def clean_positions(positions):
    """
    Fonction de nettoyage des positions enregistrées au cours de l'expérience. Pour rappel, une valeur de -1, indique que le sujet n'a pas été détecté par le réseau de neurones.
    :param positions: 
    :return: 
    """
    y = np.where(positions == -1)[0]
    diff_y = np.diff(y)
    diff_y = np.vstack((np.arange(1, len(diff_y) + 1), diff_y)).T
    k = 0
    begin = 0
    for i, elt in diff_y:
        if elt != 1:
            positions[y[i - 1]] = positions[y[i - 1] + 1]
            positions[y[i]] = positions[y[i] - 1]
            if k != 0:
                end = y[i - 1]
                filler = np.full(shape=k, fill_value=positions[begin - 1])
                positions[begin:end] = filler
                k = 0
        else:
            if k == 0:
                begin = y[i - 1]
            k += 1
    remainder = np.where(positions == -1)[0]
    if len(remainder) != 0 and k != 0:
        pass

    return positions



# Charger les positions
positions = rs.load_positions_file(path)
positions = [file for file in positions if file != ""]
p = positions[0]

# Lire les positions à partir du fichier bin
p = np.fromfile(p, dtype=np.int32)
x_p = p[np.arange(0, len(p), step=2)]  # Positions x
y_p = p[np.arange(1, len(p), step=2)]  # Positions y

# Nettoyage des positions pour retirer les valeurs -1 (si nécessaire)
clean_x_p = clean_positions(x_p)
clean_y_p = clean_positions(y_p)

# Calcul du nombre de frames pour l'animation
n_frames = len(clean_x_p)



# Création de la figure et des axes
fig, ax = plt.subplots()
ax.set_xlim(np.min(clean_x_p) - 10, np.max(clean_x_p) + 10)  # Limites de l'axe x ajustées
ax.set_ylim(np.min(clean_y_p) - 10, np.max(clean_y_p) + 10)  # Limites de l'axe y ajustées
ax.set_title("Animation des positions (x, y) au cours du temps")

# Initialisation du tracé
line, = ax.plot([], [], 'r-', label="Trajectoire")  # Trajectoire (ligne rouge)
point, = ax.plot([], [], 'bo', label="Position")  # Point animé (cercle bleu)
ax.legend()

# Fonction d'initialisation (avant le début de l'animation)
def init():
    point.set_data([], [])  # Aucune position au départ
    line.set_data([], [])   # Trajectoire vide au départ
    return point, line

# Fonction de mise à jour pour chaque frame de l'animation
def animate(i):
    # Mise à jour de la trajectoire (toutes les positions précédentes)
    line.set_data(clean_x_p[:i+1], clean_y_p[:i+1])

    # Mise à jour de la position du point (coordonnées x et y)
    point.set_data(clean_x_p[i], clean_y_p[i])
    
    return point, line

# Création de l'animation
ani = animation.FuncAnimation(fig, animate, frames=n_frames, init_func=init, blit=False, interval=0.0000000001)

# Affichage de l'animation
plt.show()
