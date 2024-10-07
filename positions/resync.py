import numpy as np
import os
from glob import glob
import re
import matplotlib.pyplot as plt


class Mapping(object):
    """

    """
    def __init__(self, width, n_freq, mid, octave):
        """
        Construction d'un objet de Mapping.
        :param width: Largeur de l'image en pixels.
        :param n_freq: Nombre de fréquences.
        :param mid: Fréquence du milieu.
        :param octave: Nombre d'octaves.
        """
        self.bandwidth = width / (n_freq - 1)
        self.half_bandwidth = self.bandwidth // 2
        self.width = width
        self.mid = mid
        self.o = octave
        self.m_numFrequency = n_freq
        self._lut_indices = np.zeros(self.width, dtype=int)
        self.tones = np.zeros(n_freq)
        self._lut_tones = np.zeros(self.width)
        self._build_lut()

    def _build_lut(self):
        """
        Construit la "look-up table" des indices du mapping et également la LUT des fréquences.
        """

        def mapping(mid, n, o):
            _t = np.zeros(n)
            m_idx = n // 2
            _t[m_idx] = 0
            s = o / n
            _t[:m_idx] = np.arange((- n // 2) + 1, 0)
            _t[m_idx + 1:] = np.arange(1, n // 2 + 1)
            _t = np.round(mid * np.power(2, _t * s))
            return _t

        def func(position):
            if position < self.half_bandwidth:
                index = 0
            elif position > (self.width - self.half_bandwidth):
                index = self.m_numFrequency - 1
            else:
                index = position - self.half_bandwidth
                index //= self.bandwidth
                index += 1
            return int(index)

        def func_fill_tones(position, tones):
            return tones[func(position)]

        self.tones = mapping(self.mid, self.m_numFrequency, 7.0)
        for i in range(self.width):
            self._lut_indices[i] = func(i)
            self._lut_tones[i] = func_fill_tones(i, self.tones)

    def get_start_stop(self, motion):
        """
        Renvoie les indices de départ et d'arrivée pour un mouvement donnée.
        """
        # print(motion)
        start = self._lut_indices[motion[0]]
        stop = self._lut_indices[motion[-1]]
        return start, stop

    def get_freq_start_stop(self, motion):
        start = self._lut_tones[motion[0]]
        stop = self._lut_tones[motion[-1]]
        return start, stop

    def convert_to_frequency(self, motion):
        """
        Renvoie les fréquences correspondantes aux positions dans un vecteur.
        :param motion:
        :return:
        """
        t = np.zeros(len(motion), dtype=float)
        for i, _p in enumerate(motion):
            t[i] = self._lut_tones[_p]
        return t


def clean_positions(positions):
    """
    Fonction importante.
    Fonction de nettoyage des positions enregistrées au cours de l'expérience. Pour rappel,
    une valeur de -1, indique que le sujet n'a pas été détecté par le réseau de neurones.
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


def stack_positions_data(positions_files):
    """
    Retourne les vecteurs continus, et un np.ndarray des positons où il faudra recouper pour séparer à nouveau.
    :param positions_files:
    :return:
    """
    # Ici, il va falloir garder en mémoire la dimension exacte de chaque sous tableau.
    split_mem = np.zeros(len(positions_files) - 1, dtype=int)
    is_playback = list()
    playback_pattern = "positions_Playback_playback_0[0-9]"
    positions_vectors = list()
    start = 0
    for i, file in enumerate(positions_files):
        # Je connais les indices où sont les positions de playback.
        if re.search(playback_pattern, file):
            is_playback.append(i)
        vec = np.fromfile(file, dtype=np.int32)
        if i != (len(positions_files) - 1):
            start += len(vec)
            split_mem[i] = start
        positions_vectors.append(vec)
    # print(is_playback)
    cp = clean_positions(np.hstack(positions_vectors))
    cp = np.split(cp, split_mem)
    playback_positions_list = [cp[i] for i in is_playback]
    return playback_positions_list


def detect_frequency_switch(vec, mapping, mock_tones):
    """
    Fonction qui a pour objectif de détecter les changements de fréquences.
    :param mapping:
    :param vec:
    :param mock_tones:
    :return:
    """
    # print(np.unique(vec))
    tone_vec = mapping.convert_to_frequency(vec)
    d = np.diff(tone_vec)
    idx = np.where(d != 0)[0] + 1
    switch = tone_vec[idx]
    return switch


def load_mock_tones_files(folder):
    # mock_pattern = "tracking_mock_0[0-9]"
    glob_files = glob(os.path.join(folder, "tones", "tracking_mock_*.bin"))
    mock_files = ["" for _ in glob_files]
    for file in glob_files:
        match = re.search(r"tracking_mock_(\d+)", file)
        if match:

            value = int(match.group(1))
            p = np.fromfile(file, dtype=float)
            mock_files[value] = p

    return mock_files


def load_playback_tones_files(folder):
    # mock_pattern = "tracking_mock_0[0-9]"
    glob_files = glob(os.path.join(folder, "tones", "playback_*.bin"))
    mock_files = ["" for _ in glob_files]
    for file in glob_files:
        match = re.search(r'playback_(\d+)', file)
        if match:
            value = int(match.group(1))
            p = np.fromfile(file, dtype=float)
            mock_files[value] = p
    return mock_files


def load_positions_file(folder):
    """
    Retourne une liste avec les fichiers des positions enregistrées au cours de l'expérience ordonnés.
    :param folder: Dossier de sauvegarde de l'expérience.
    :return: Une liste avec les noms de fichiers dans l'ordre chronologique.
    """
    # Pipeline : extraire tous les fichiers, les remettre dans l'ordre. Nettoyer, couper.
    # Extraire les fréquences du Mock.
    # Comparer la longueur, une fois les changements détectés. OK.
    pause_pattern = r"positions_Pause_0[0-9]"
    playback_pattern = r"positions_Playback_playback_0[0-9]"
    tracking_pattern = r"positions_tracking_0[0-9]"
    warmup_pattern = r"positions_Playback_warmup_0[0-9]"

    glob_files = glob(os.path.join(folder, "positions", "positions_*.bin"))
    types_pos_list = [list() for _ in range(4)]
    for file in glob_files:
        if re.search(pause_pattern, file):
            types_pos_list[0].append(file)
        elif re.search(warmup_pattern, file):
            types_pos_list[1].append(file)
        elif re.search(r"positions_tracking_0[0-9]", file) is not None:
            types_pos_list[2].append(file)
        elif re.search(playback_pattern, file):
            types_pos_list[3].append(file)

    out = ["" for _ in glob_files]

    for file_name in types_pos_list[0]:
        match = re.search(r"positions_Pause_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            if value == 0:
                out[0] = file_nameidx = 2
            else:
                out[-1] = file_name

    for file_name in types_pos_list[1]:
        match = re.search(r"positions_Playback_warmup_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            if value == 0:
                out[1] = file_name
            else:
                out[-2] = file_name
    idx = 2
    n_iter = len(types_pos_list[2])

    for i in range(n_iter):
        file_name = types_pos_list[2][i]
        match = re.search(r"positions_tracking_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            print(f"TR : {value}")
            out[idx + value * 2] = file_name

        file_name = types_pos_list[3][i]
        match = re.search(r"positions_Playback_playback_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            print(f"PB : {value}")

            out[idx + value * 2 + 1] = file_name
    # for i in range(len(types_pos_list[2])):
    #     out[idx] = types_pos_list[2][i]  #  [, f"Tracking_0{i}"]  # Tracking
    #     out[idx + 1] = types_pos_list[3][i]  # , f"Playback_0{i}"]  # Playback
    #     idx += 2
    return out


def test(folder):
    o = load_positions_file(folder)
    split_mem = np.zeros(len(o) - 1, dtype=int)
    positions_vectors = list()
    start = 0
    for i, file in enumerate(b):
        # Je connais les indices où sont les positions de playback.
        vec = np.fromfile(file, dtype=np.int32)
        if i != (len(o) - 1):
            start += len(vec)
            split_mem[i] = start
        positions_vectors.append(vec)
    cp = np.hstack(positions_vectors)
    cp = np.split(cp, split_mem)
    print(f"Length : {len(cp)}, {len(o)}, {len(positions_vectors)}")
    for i in range(len(cp)):
        print(len(cp[i]) == len(positions_vectors[i]))
    for i in range(len(cp)):
        print(cp[i] == positions_vectors[i])

    return


def frequency_correction(folder):
    """

    :param folder:
    :return:
    """
    # C'est tout le pipeline de correction des fréquences.
    # 0. Créer le mapping
    mapping = Mapping(1920, 33, 2000., 7)

    # 1. Charger les "mock tones".
    mock_tones = load_mock_tones_files(folder)
    #
    positions = load_positions_file(folder)
    clean_pb_pos = stack_positions_data(positions)
    mock_corrected = [detect_frequency_switch(vec, mapping, mock_tones[i]) for i, vec in enumerate(clean_pb_pos)]
    # Placer chaque changement de fréquence, avec la position correspondante.
    # Trigger -> Indice
    # Extraire le tableau de positions entre deux indices, et les répartir "evenly".
    return mock_tones, positions, clean_pb_pos, mock_corrected


if __name__ == "__main__":
    a, b, c, e = frequency_correction(
        "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230218/MUROLS_20230218_SESSION_01")
