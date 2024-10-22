"""
TODO : incorporer la 2D.
"""
import numpy as np


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


def clean_positions_xy(positions, triggers=None):
    """

    :param positions:
    :param triggers:
    :return:
    """
    x = np.array([])
    y = np.array([])
    if triggers is None:
        assert (len(x) == len(y)), "x and y should be equals."
    else:
        assert (len(x) == len(y) == len(triggers)), "triggers, x and y should be equal."

    clean_x = clean_positions(x)
    clean_y = clean_positions(y)

    pass


def load(folder, has_triggers=False, is_2d=False):
    # Charger les données, les réordonner, le fait que ce soit 2D vient du .json de manip, pareil pour les triggers.

    # On va dans le dossier de positions, on les mets dans l'ordre, via le .json ?
    pass