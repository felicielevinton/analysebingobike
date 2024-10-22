import numpy as np
import os
from Experience.experiment_type import get_pattern_from_type, get_type_from_pattern


class PT(object):
    def __init__(self, positions, triggers):
        assert (len(positions) == len(triggers)), "Tones and Triggers have different length."
        self.length = len(positions)
        self.tones = positions
        self.triggers = triggers


class PositionPair(object):
    def __init__(self, values, triggers, type_of, number=None, order=None):
        assert (len(values) == len(triggers)), "Tones and Triggers have different length."
        self.values = values

        self.triggers = triggers
        self.tt = PT(values, triggers)

        assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "PureTones",
                            "silence"]), "Wrong type..."
        self.type = type_of

        if order is not None:
            self.order = order
        else:
            self.order = None

        if number is not None:
            self.number = number
            self.pattern = get_pattern_from_type(self.type) + str(self.number)
        else:
            self.number = None
            self.pattern = None

    def get_stacked(self):
        return np.vstack((self.values, self.triggers))

    def get_values(self):
        return self.values

    def get_triggers(self):
        return self.triggers

    def get_pairs(self):
        return self.tt

    def get_pattern(self):
        return self.pattern

    def get_type(self):
        return self.type

    def get_begin_and_end_triggers(self):
        return self.triggers[0], self.triggers[-1]


class SequencePT(object):
    """

    """
    def __init__(self, folder=None, n_iter=None):
        self._fn = "pt.npz"
        self.container = dict()
        self.keys = list()
        self.order = np.empty(0, dtype=int)
        self.numbers = np.empty(0, dtype=int)
        self.total_iter = 0
        self.allowed = ["playback", "tracking", "warmup", "warmdown"]
        if n_iter is not None:
            self.total_iter = n_iter
        self.recording_length = 0

        self._movement = None
        self._separator = None
        if folder is not None:
            self._load(folder)

    def add(self, pairs):
        pattern = pairs.get_pattern()
        order = pairs.order
        number = pairs.number
        assert (pattern not in self.keys), "Already in DataStructure."
        assert (order not in self.order), "Already in DataStructure."
        self.numbers = np.hstack((self.numbers, number))
        self.order = np.hstack((self.order, order))
        self.keys.append(pattern)
        self.container[pattern] = pairs

    def _load(self, folder):
        d = np.load(os.path.join(folder, self._fn), allow_pickle=True)
        self.recording_length = d["recording_length"][0]
        self.order = d["order"]
        self.total_iter = d["n_iter"][0]
        self.keys = [key.decode() for key in d["keys"]]
        self.numbers = d["numbers"]
        for i, key in enumerate(self.keys):
            positions, triggers = d[key][0], d[key][1]
            self.container[key] = PositionPair(positions, triggers,
                                               type_of=get_type_from_pattern(key),
                                               order=self.order[i])

    def get_number_iteration(self):
        return self.total_iter

    def _foo(self):
        """
        La méthode doit diviser le mouvement total en segments de mouvements / repos.
        :return:
        """
        for pair in self.container.values():
            pass
        pass

    def save(self, folder):
        """

        """
        fn = os.path.join(folder, self._fn)
        kwargs = dict()
        kwargs["order"] = np.array(self.order)
        kwargs["n_iter"] = np.array([self.total_iter])
        kwargs["recording_length"] = np.array([self.recording_length])
        kwargs["keys"] = self._build_chararray()
        kwargs["numbers"] = self.numbers
        for key in self.container.keys():
            kwargs[key] = self.container[key].get_stacked()
        np.savez(fn, **kwargs)

    def _build_chararray(self):
        n = np.array(self.keys).shape
        ch = np.chararray(n, itemsize=5)
        for i, elt in enumerate(self.keys):
            ch[i] = elt
        return ch

    def get_xp_number(self, type_of, n):
        """
        On demande une expérience d'un type donné, à un moment donné.
        """
        assert (type_of in self.allowed), "Wrong type..."
        if type_of not in ["warmup", "warmdown"]:
            assert (n < self.total_iter), "Unavailable."
        pattern = get_pattern_from_type(type_of) + str(n)
        assert (pattern in self.keys), "Not existing"
        return self.container[pattern]