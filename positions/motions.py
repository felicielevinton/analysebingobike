import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
"""
Comment organiser tout ça ?
* 1. découpage entre Pauses et Mouvements
* 2. Découpage d'un mouvement en sous mouvement. 
* 3. Quels objets ?

Le gros objectif maintenant c'est de maintenir la synchronisation mouvements / triggers.
"""


def sort_chrono(segment_list):
    return sorted(segment_list, key=lambda x: x[0])


def check_segment_length(sub_vector, opposite_cat_list, min_length):
    segment_list = list()
    for elt in sub_vector:
        if len(elt) > min_length:
            segment_list.append(elt)
        else:
            opposite_cat_list.append(elt)

    return segment_list, opposite_cat_list


def divide_in_segments(p, threshold, triggers=None):
    """
    Divise les positions en "Mouvement" et "Pause".
    :param p: Vecteur de positions.
    :param threshold: Seuil de vitesse
    :param triggers:
    :return: Endroits qui sont possiblement des pauses. Cette fonction retourne des indices.
    """
    dp = np.diff(p)  # Obtenir la vitesse.
    putative_steady = np.where((dp >= -threshold) & (dp <= threshold))[0]  # valeurs comprises entre deux bornes
    triggers_to_keep = triggers[putative_steady + 1]
    # Identifier les différents segments. Et les séparer quand la différence est supérieure à 1.
    # Todo : faire quelque chose avec les triggers.
    d_put_steady = np.diff(putative_steady)
    d_cut = np.where(d_put_steady != 1)[0] + 1
    # Indices des différents segments
    out = np.split(putative_steady, d_cut)
    return out


def divide_in_segments_2(p, threshold_speed, min_length_pause, min_length_motion, triggers=None):
    """

    :param p: Vecteur de positions.
    :param threshold_speed: Seuil de vitesse
    :param min_length_pause: Longueur minimale d'un segment pour être considéré comme une pause.
    :param min_length_motion: Longueur minimale d'un segment pour être considéré comme un mouvement.
    :param triggers:
    :return: Endroits qui sont possiblement des pauses. Cette fonction retourne des indices.
    """
    # TODO : faire gaffe à la gestion des indices.
    dp = np.diff(p)  # Obtenir la vitesse.
    indices = np.arange(len(dp), dtype=int)
    if triggers is None:
        triggers = np.arange(len(p), dtype=int)

    # Important de garder l'ordre.
    # putative_steady est un vecteur de booléens.
    putative_steady = np.logical_and(threshold_speed <= dp, threshold_speed >= dp)
    # D'abord, travailler sur la taille des segments de pauses.
    change_indices = np.where(np.diff(putative_steady))[0] + 1
    motion_change_indices = np.where(np.diff(~putative_steady))[0] + 1
    # Ici, j'ajoute l'indice de début de la liste et l'indice de fin de la liste.
    slices = np.concatenate(([0], change_indices, [len(putative_steady)]))
    motion_slices = np.concatenate(([0], motion_change_indices, [len(putative_steady)]))
    # Ici, je découpe en sous tableaux :
    # sub_vectors = [putative_steady[slices[i]:slices[i+1]] for i in range(len(slices)-1) if putative_steady[slices[i]]]

    sub_vectors_idx_steady = [indices[slices[i]:slices[i+1]]
                              for i in range(len(slices)-1)
                              if putative_steady[slices[i]]]

    sub_vectors_idx_motions = [indices[motion_slices[i]:motion_slices[i+1]]
                               for i in range(len(motion_slices) - 1)
                               if putative_steady[motion_slices[i]]]

    # Maintenant, j'élimine les sous tableaux trop courts, et les place dans la catégorie opposée.
    real_pauses, sub_vectors_idx_motions = check_segment_length(sub_vectors_idx_steady,
                                                                sub_vectors_idx_motions,
                                                                min_length_pause)
    real_motions, real_pauses = check_segment_length(sub_vectors_idx_motions,
                                                     real_pauses,
                                                     min_length_motion)

    # Je trie la liste en fonction de la valeur du premier indice de la liste. Passer à la fin.
    real_motions = sort_chrono(real_motions)
    real_pauses = sort_chrono(real_pauses)

    # Dernière étape sera le merge.
    # Faire np.diff : va gérer les fusions puis séparer avec "np.split".
    real_motions = np.hstack(real_motions)
    wm = np.where(np.diff(real_motions))[0] + 1
    real_motions = np.split(real_motions, wm)
    real_pauses = np.hstack(real_pauses)
    wp = np.where(np.diff(real_pauses))[0] + 1
    real_pauses = np.split(real_pauses, wp)

    # Sortie : Créer un container d'objets Pause et de Motions.
    # S'assurer que toute la sortie est organisée correctement.
    # Todo : faire quelque chose avec les triggers.
    # d_put_steady = np.diff(putative_steady)
    # d_cut = np.where(d_put_steady != 1)[0] + 1
    # Indices des différents segments
    # out = np.split(putative_steady, d_cut)
    return


def process_positions(positions_array, triggers, threshold=3, duration_threshold=0.5, framerate=30):
    """
    Recherche des moments (à partir des positions enregistrées) où le sujet effectue une pause. Nous recherchons des
    moments où la vitesse est nulle (à +/- threshold pixels près) pendant une significative.
    :param positions_array: Vecteur de positions.
    :param triggers: Vecteur de temps de triggers.
    :param threshold: Seuil pour le maximum de vitesse (en différence de pixels) à rechercher pour définir une pause.
    :param duration_threshold: Durée minimale, à une vitesse quasiment nulle pour que l'on considère un arrêt.
    :param framerate: Fréquence d'acquisition de la caméra.
    :return:
    """
    assert (len(triggers) == len(positions_array)), "Triggers and positions must have equal length."
    n_frame = int(duration_threshold * framerate)
    steady_segments = list()
    motions_indices = list()

    out = divide_in_segments(positions_array, threshold, triggers)  # Forcer l'entrée des triggers ici ?

    # Ici, je regarde si les segments de basse vitesse ont une longueur supérieure au seuil défini.
    for segment in out:
        if len(segment) > n_frame:
            steady_segments.append(segment)

    stack_idx_steady = np.hstack(steady_segments)  # ?

    # Très inélégant...
    for i, _ in enumerate(positions_array):
        if i not in stack_idx_steady:
            motions_indices.append(i)

    motions_indices = np.array(motions_indices)
    d_put_motions = np.diff(motions_indices)  # Je recoupe là où la différence d'indice est différente de 1.
    d_cut = np.where(d_put_motions != 1)[0] + 1
    motions_segments = np.split(motions_indices, d_cut)
    clean_motion_segments = list()

    # Fusionner les courts segments de mouvement dans les segments steady
    for segment in motions_segments:
        if len(segment) < 9:  # Si le segment est court
            start_time = segment[0]
            # Trouver l'index du segment steady où ce segment de mouvement court doit être inséré
            idx_to_insert = next((i for i, s in enumerate(steady_segments) if s[-1] >= start_time), None)
            if idx_to_insert is not None:
                # Fusionner ce segment de mouvement avec le segment steady correspondant
                steady_segments[idx_to_insert] = np.concatenate((steady_segments[idx_to_insert], segment))
        else:
            clean_motion_segments.append(segment)

    # TODO : Garder la trace de l'organisation des segments.
    steady = [np.vstack((segment, positions_array[segment])) for segment in steady_segments]
    moving = [np.vstack((segment, positions_array[segment])) for segment in clean_motion_segments if len(segment) > 5]
    return moving, steady


def zero_crossings(arr):
    """
    Fonction pour le processing des positions.
    Détecte si un signal change de signe (c'est-à-dire : passe par zéro).
    :param arr: Signal.
    :return:
    """
    # Calcul de la différence de signe entre les éléments consécutifs
    d_arr = np.diff(arr)
    sign_change = np.sign(d_arr[:-1]) * np.sign(d_arr[1:])
    # Les endroits où sign_change est négatif sont des passages par zéro
    crossings = np.where(sign_change < 0)[0]
    closest_indices = []
    for index in crossings:
        # Indices des éléments avant et après le passage par zéro
        before, after = index, index + 1
        # Trouver l'élément le plus proche de zéro entre les deux
        closest_index = before if abs(d_arr[before]) < abs(d_arr[after]) else after
        closest_indices.append(closest_index)

    return np.array(closest_indices)


def motion_cleaning(motion, turns, indices, framerate):
    """
    Fonction pour le processing des positions.

    Nettoie la liste de segments de mouvements en fusionnant ceux qui sont trop courts.
    :param motion:
    :param turns: Liste de np.array représentant les segments de mouvement
    :param indices: Indices originaux de ces segments dans le tableau de mouvement d'origine.
    :param framerate: Fréquence de capture de la caméra.
    :return: Liste nettoyée de segments de mouvements et leurs indices associés.
    """
    clean_turns = list()
    clean_indices = list()
    current_merge = list()  # Liste temporaire pour fusionner les segments trop courts
    n_turns = len(indices)  # Nombre de virages.
    threshold = int(framerate / 5)  # Seuil de longueur en dessous duquel un segment est considéré comme "trop court"
    too_short = False
    for i, t in enumerate(turns):
        can_add = i < n_turns  # seulement ajouter si i est plus petit que n_turns
        if len(t) > threshold:
            if too_short:
                too_short = False  # Réinitialiser le drapeau
                new_length = sum([len(ts) for ts in current_merge])
                if new_length > threshold:
                    clean_turns.append(np.hstack(current_merge))
                    clean_indices.append(i - 1)
                    clean_turns.append(t)
                else:
                    current_merge.append(t)
                    clean_turns.append(np.hstack(current_merge))
                current_merge = list()  # vider current_merge pour le prochain groupe
            else:
                clean_turns.append(t)
        else:
            too_short = True  # activer le drapeau
            current_merge.append(t)  # ajouter à la liste pour potentiellement fusionner plus tard

        if can_add and not too_short:
            clean_indices.append(i)
    if too_short:
        clean_turns.append(np.hstack(current_merge))

    if len(clean_indices) > 0:
        clean_indices = np.array([indices[i] for i in clean_indices])
        turns = np.split(motion, clean_indices)
    else:
        turns = [motion]
    return turns, clean_indices


def find_turns(motion, threshold=2, framerate=30):
    """
    Fonction pour le processing des positions.

    Trouve les virages dans le mouvement, redivise en courts Segments.
    :param motion:
    :param threshold:
    :param framerate:
    :return:
    """
    # Recevoir des objets motions et le diviser en segments entre chaque virage.
    if len(motion) < 5:
        print("ERROR", len(motion))

    if len(motion) < int(framerate / 3):
        return [motion], list()

    smooth_motion = mean_smoothing(motion, size=int(framerate / 6), pad_size=framerate)  # Lissage pour le bruit.
    diff_smooth_motion = np.diff(smooth_motion)  # Vitesse.
    turns = divide_in_segments(smooth_motion, threshold, triggers=None)  # Détection des virages.
    points = list()
    original_indices = list()  # pour stocker les indices argmin dans le tableau original m_pos

    for turn in turns:
        zc = zero_crossings(diff_smooth_motion[turn])
        if len(zc) > 0:
            original_index = turn[zc]  # convertit l'indice relatif en indice dans m_pos
            points.append(zc)
            original_indices.extend(original_index)

    original_turns = np.split(smooth_motion, original_indices)
    turns, indices = motion_cleaning(motion, original_turns, original_indices, framerate=framerate)
    return turns, indices


def gaussian_smoothing(x, sigma=5, size=10, pad_size=None):
    """
    Créer un kernel gaussien
    M = taille de la fenêtre.
    std = distribution de la gaussienne.
    """
    kernel = signal.windows.gaussian(M=size, std=sigma)
    return smooth(kernel, x, pad_size)


def mean_smoothing(x, size=10, pad_size=None):
    """
    Créer un noyau gaussien
    M = taille de la fenêtre.
    std = distribution de la gaussienne.
    """
    kernel = np.ones(size) / size
    return smooth(kernel, x, pad_size)


def smooth(kernel, x, pad_size):
    """
    Fonction générique pour le lissage.
    :param kernel: Noyau pour la convolution
    :param x: Signal à lisser.
    :param pad_size: Ajout de valeurs aux extrémités pour éviter les artefacts.
    :return: Signal lissé.
    """
    if pad_size is not None:
        x = np.hstack([np.full(pad_size, x[0]), x, np.full(pad_size, x[-1])])
        x_conv = signal.fftconvolve(x, kernel, "same")[pad_size:-pad_size]
    else:
        x_conv = signal.fftconvolve(x, kernel, "same")

    return x_conv


