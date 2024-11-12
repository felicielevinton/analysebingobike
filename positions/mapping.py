import numpy as np


# todo : ajouter une fonction read_mapping dans le fichier .json d'expérience.

def read_mapping_json(d_json):
    # On connait la largeur de l'image.
    # Nombre de fréquences, la fréquence du milieu, le nombre d'octaves.
    pass


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

    def convert_to_index(self, f):
        return np.where(self.tones == f)[0][0]

    def convert_to_octaves(self, f):
        return np.log2(f / self.mid)