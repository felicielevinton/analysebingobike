from motions import find_turns, process_positions
import numpy as np

# AUCUN INTERET
class Status(object):
    """

    """

    def __init__(self, ):
        self.x = 0
        self.t = 0
        self.moving = False
        self.speed = 0

        pass


class MotionAnalyzer(object):
    def __init__(self):
        # Avoir tous les triggers de positions.
        pass

    def from_time_get_status(self, t):
        """
        L'idée, c'est de passer un temps, et de récupérer la position du furet, s'il est engagé
        dans un mouvement ou non.
        :param t:
        :return:
        """
        pass


class MotionContainer(object):
    """
    Objet qui permettra d'utiliser les positions lors des analyses.

    Que lui passer ? Que faire ? Quelles informations ?
    """
    def __init__(self):
        self._n_pauses = 0
        self._n_motions = 0
        self._container = dict()
        # Ajouter un moyen de sauvegarder les balises.
        # Sauvegarde dans un HDF5.
        pass

    def add(self, segment, type_of):
        if type_of == "Pause":
            num_str = self._convert_num_to_str(self._n_pauses)
            tag = "p" + num_str
            self._n_pauses += 1
        else:
            num_str = self._convert_num_to_str(self._n_motions)
            tag = "m" + num_str
            self._n_motions += 1
        self._container[tag] = segment

    def get(self, type_of):
        pass

    def from_time_get_status(self, t):
        """
        L'idée, c'est de passer un temps, et de récupérer la position du furet, s'il est engagé
        dans un mouvement ou non.
        :param t:
        :return:
        """
        for segment in self._container.items():
            if segment.contains(t):
                break

    @staticmethod
    def _convert_num_to_str(n):
        num_str = str(n)
        if n < 10:
            num_str = "00" + num_str
        elif n < 100:
            num_str = "0" + num_str
        return num_str


class PositionSegment(object):
    """
    Classe abstraite. Pourra être de type "Pause" ou de type "Motion".
    """
    def __init__(self, x, t, framerate, mapping):
        self.x = x
        self.t = t
        self.framerate = framerate
        self.mapping = mapping
        self.duration = len(self.x) / self.framerate
        self.max_speed = None
        self.distance = None
        self.start, self.stop = None, None
        self.start_f, self.stop_f = None, None

    def contains(self, t):
        v = True if t in self.t else False
        return v


class Pause(PositionSegment):
    def __init__(self, x, t, framerate, mapping):
        super().__init__(x, t, framerate, mapping)


class Motion(PositionSegment):
    """
    Changer de nom pour la classe.
    """

    def __init__(self, x, t, framerate, mapping):
        """

        :param x: Vecteur positions.
        :param t: Vecteur de temps de triggers.
        :param framerate: Fréquence d'acquisition de la caméra.
        :param mapping: Carte des fréquences de l'expérience.
        """
        super().__init__(x, t, framerate, mapping)
        self.dx = np.diff(self.x)
        self.mapping = mapping
        self.max_speed = self.dx[np.argmax(np.abs(self.dx))]
        self.distance = np.sum(np.abs(self.dx))
        self.start, self.stop = self.mapping.get_start_stop(self.x)
        self.start_f, self.stop_f = self.mapping.get_freq_start_stop(self.x)
        # todo : Passer les triggers ici.
        self.sub_motion = SubMotion(self.x, self.framerate, threshold=2)

    def get_max_speed(self):
        return self.max_speed

    def get_distance_covered(self):
        return self.distance

    def get_start_stop(self):
        return self.start, self.stop

    def get_duration(self):
        return self.duration

    def get_submotion(self):
        return

    def get_num_switches(self):
        tones = self.mapping.convert_to_frequency(self.x)
        _, c = np.unique(tones, return_counts=True)
        # d_tones = np.diff(tones)
        return sum(c)

    def get_n_tones(self):
        tones = self.mapping.convert_to_frequency(self.x)
        return len(np.unique(tones))

    def get_features(self):
        direction = 1 if self.x[0] > self.x[-1] else -1
        if self.start > self.stop:
            stop = self.start
            start = self.stop
        else:
            start = self.start
            stop = self.stop

        # mettre
        foo = np.diff(self.dx)
        if len(foo) < 1:
            print(2)
        # print())
        return np.array([self.max_speed, self.distance, self.duration,
                         self.dx.mean(), np.abs(self.dx).mean(), np.mean(self.x), np.var(self.x),
                         self.stop_f, self.start_f,
                         start, stop,
                         np.log2(self.stop_f / self.start_f),
                         self.get_n_tones(),
                         self.get_num_switches(),
                         direction
                         ])


class SubMotion(object):
    def __init__(self, x, framerate=30, threshold=2):
        self.framerate = framerate
        self.threshold = threshold
        find_turns(x, framerate=self.framerate, threshold=threshold)
        pass

    def get_max_speed(self):
        return self.max_speed

    def get_distance_covered(self):
        return self.distance

    def get_start_stop(self):
        return self.start, self.stop

    def get_duration(self):
        return self.duration

    def get_n_turns(self):
        pass