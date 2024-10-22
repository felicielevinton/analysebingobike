import numpy as np
import os
from glob import glob
import re
import Experience.get_data as gd
from Experience.get_good_sessions import get_good_sessions
from Experience.Mapping import Mapping
import matplotlib.pyplot as plt
from copy import deepcopy
from PTSequence_Experimental import PT, PositionPair, SequencePT


def clean_triggers(triggers):
    """
    Fonction de nettoyage des positions enregistrées au cours de l'expérience. Pour rappel, une valeur de -1, indique
    que le sujet n'a pas été détecté par le réseau de neurones.
    :param triggers:
    :return:
    """
    zeroes = np.equal(triggers, 0)
    changes = np.where(np.diff(np.concatenate(([False], zeroes, [False]))))[0]
    segments = np.split(triggers, changes)

    first_segment = np.empty(0)
    last_segment = np.empty(0)
    if len(segments[0]) == 0:
        segments = segments[1:]

    if np.all(np.equal(segments[0], 0)):
        first_segment = segments[0]
        segments = segments[1:]

    if len(segments[-1]) == 0:
        segments = segments[:-1]

    if np.all(np.equal(segments[-1], 0)):
        last_segment = segments[-1]
        segments = segments[:-1]

    for i in np.arange(1, len(segments), step=2):
        segment = segments[i]
        assert(np.all(np.equal(segment, 0)))
        n = segment.size
        t0 = segments[i - 1][-1]
        t1 = segments[i + 1][0]

        dt = int((t1 - t0) / (n + 1))
        if len(segment) > 1:
            segments[i] = np.full_like(segment, t0) + np.arange(1, 1 + len(segment), dtype=int) * dt
            assert(sum(np.equal(segments[i], 0)) == 0)
        elif len(segment) == 1:
            segments[i] = np.array([t0 + dt])
    triggers = np.hstack(segments)

    triggers = np.hstack([first_segment, triggers, last_segment])
    dt = None
    i_last = None
    t1 = None

    if first_segment.size > 0:
        remaining_zeroes = np.where(triggers != 0)[0]
        i_first = remaining_zeroes[0]
        i_last = remaining_zeroes[-1]
        segment = triggers[:i_first]
        dt = int(np.diff(triggers[i_first:i_last]).mean())
        t0 = triggers[i_first]
        t1 = triggers[i_last]
        to_fill = segment[:i_first]
        triggers[:i_first] = np.full_like(segment, t0) - np.arange(len(to_fill))[::-1] * dt

    if last_segment.size > 0:
        if dt is None:
            remaining_zeroes = np.where(triggers != 0)[0]
            i_last = remaining_zeroes[-1]
            dt = int(np.diff(triggers[:i_last]).mean())
            t1 = triggers[i_last]
        segment = triggers[i_last:]
        triggers[i_last:] = np.full_like(segment, t1, dtype=int) + np.arange(len(segment), dtype=int) * dt

    return triggers


def check_forced_push(pos1, pos2, tones, mapping):
    m1 = np.full_like(pos1, True, dtype=bool)
    m2 = np.full_like(pos2, False, dtype=bool)
    m = np.hstack((m2, m1))
    pos = np.hstack((pos2, pos1))
    sw, _ = detect_frequency_switch(pos, mapping, m)
    forced_push = np.array_equal(tones, sw)  # Si faux : forced_push.

    return


def clean_positions(positions):
    """
    Fonction de nettoyage des positions enregistrées au cours de l'expérience. Pour rappel, une valeur de -1, indique
    que le sujet n'a pas été détecté par le réseau de neurones.
    :param positions:
    :return:
    """
    if positions[0] == -1:
        first = np.where(positions > -1)[0][0]
        val = positions[first]
        positions[:first] = np.full_like(positions[:first], val)

    if positions[-1] == -1:
        last = np.where(positions > -1)[0][-1]
        val = positions[last]
        positions[last:] = np.full_like(positions[last:], val)
    y = np.where(positions == -1)[0]
    # indices = np.arange(len(positions))
    minus_one = np.equal(positions, -1)
    changes = np.where(np.diff(np.concatenate(([False], minus_one, [False]))))[0]
    segments = np.split(positions, changes)

    for i in np.arange(1, len(segments), step=2):
        segment = segments[i]
        assert(np.all(np.equal(segment, -1)))
        if len(segment) == 1:
            segments[i] = np.array([segments[i-1][-1]])
        else:
            n = len(segment)
            first = segments[i-1][-1]
            last = segments[i+1][0]
            step = int((last - first) / n)
            # np.arange(first, last, step=step)
            segments[i] = np.full_like(segment, first)
    positions = np.hstack(segments)
    return positions


def gather_positions_file(folder, n_blocks):
    """
    Retourne une liste avec les fichiers des positions enregistrées au cours de l'expérience ordonnés.
    :param folder: Dossier de sauvegarde de l'expérience.
    :param n_blocks: Nombre de blocks.
    :return: Une liste avec les noms de fichiers dans l'ordre chronologique.
    """
    # Pipeline : extraire tous les fichiers, les remettre dans l'ordre. Nettoyer, couper.
    # Extraire les fréquences du Mock.
    # Comparer la longueur, une fois les changements détectés. OK.
    pause_pattern = "positions_Pause_0[0-9]"
    playback_pattern = "positions_Playback_playback_0[0-9]"
    tracking_pattern = "positions_Playback_tracking_0[0-9]"
    warmup_pattern = "positions_Playback_warmup_0[0-9]"

    glob_files = glob(os.path.join(folder, "positions", "positions_*.bin"))
    types_pos_list = [list() for _ in range(4)]
    for file in glob_files:
        if re.search(pause_pattern, file):
            types_pos_list[0].append(file)
        elif re.search(warmup_pattern, file):
            types_pos_list[1].append(file)
        elif re.search(tracking_pattern, file):
            types_pos_list[2].append(file)
        elif re.search(playback_pattern, file):
            types_pos_list[3].append(file)

    out = ["" for _ in glob_files]
    types = ["" for _ in glob_files]
    nums = [0 for _ in glob_files]
    for file_name in types_pos_list[0]:
        match = re.search(r"positions_Pause_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            if value == 0:
                out[0] = file_name
                types[0] = "pause"
                nums[0] = 0
            else:
                out[-1] = file_name
                types[-1] = "pause"
                nums[-1] = 1

    for file_name in types_pos_list[1]:
        match = re.search(r"positions_Playback_warmup_(\d+)", file_name)
        if match:
            value = int(match.group(1))
            if value == 0:
                out[1] = file_name
                types[1] = "warmup"
                nums[1] = 0
            else:
                out[-2] = file_name
                types[-2] = "warmdown"
                nums[-2] = 0
    idx = 2
    # n_blocks = len(types_pos_list[2])

    for i in range(n_blocks):
        file_name = types_pos_list[2][i]
        match = re.search(r"positions_Playback_tracking_(\d+)", file_name)
        if match:
            # Ici, extraction du numéro de bloc. Je place au bon endroit dans la liste.
            value = int(match.group(1))
            pos = idx + value * 2
            out[pos] = file_name
            types[pos] = "tracking"
            nums[pos] = value

        file_name = types_pos_list[3][i]
        match = re.search(r"positions_Playback_playback_(\d+)", file_name)
        if match:
            # Ici, extraction du numéro de bloc. Je place au bon endroit dans la liste.
            value = int(match.group(1))
            pos = idx + value * 2 + 1
            out[pos] = file_name
            types[pos] = "playback"
            nums[pos] = value
    return out, types, nums


def add_position_file(file_list, fn, val, what):
    assert (what in ["tracking", "playback"])
    idx = val * 2
    if what == "playback":
        idx += 1
    file_list.append(fn)


def load_positions_file(files):
    """
    Charge les fichiers .bin et les empile.
    :param files:
    :return:
    """
    positions = list()
    seps = list()
    t = list()
    for i, file in enumerate(files):
        p = np.fromfile(file, dtype=np.int32)
        s = np.zeros_like(p)
        t.append(np.full_like(p, i))
        s[0] = 1
        positions.append(p)
        seps.append(s)
    return np.hstack(positions), np.hstack(seps), np.hstack(t)


def split(x, separators):
    """
    Sépare les différents blocs à nouveau.
    :param x:
    :param separators:
    :return:
    """
    cut_idx = np.where(separators == 1)[0][1:]
    x = np.split(x, cut_idx)
    return x


def detect_frequency_switch(vec, mapping, mask=None):
    """
    Fonction qui a pour objectif de détecter les changements de fréquences.
    :param mapping:
    :param vec:
    :param mask:
    :return:
    """
    # Je convertis le vecteur de positions en fréquences.
    tone_vec = mapping.convert_to_frequency(vec)
    d = np.diff(tone_vec)
    # Je repère les moments où les fréquences changent.
    idx = np.where(d != 0)[0] + 1
    if mask is not None:
        assert (mask.size == vec.size)
        in_mask = mask[idx]
        idx = idx[in_mask]
    # Je retourne un array de fréquences.
    switch = tone_vec[idx]
    return switch, idx


def check_size(d, tag):
    t, s = d[tag]["to"], d[tag]["sw"]
    res = 0
    if t.size > s.size:
        res = 1

    elif t.size < s.size:
        res = -1

    return res


def check_to_now_equals_sw_before(d, key_now, key_before):
    """
    Vérifie si la dernière fréquence est égale à la première du bloc suivant.
    :param d: Container
    :param key_now: Clé de l'objet en cours.
    :param key_before: Clé de l'objet précédent.
    :return: Booléen.
    """
    return d[key_now]["to"][0] == d[key_before]["sw"][-1]


def check_to_now_equals_to_before(d, key_now, key_before):
    """
    Vérifie si la dernière fréquence est égale à la première du bloc suivant.
    :param d: Container
    :param key_now: Clé de l'objet en cours.
    :param key_before: Clé de l'objet précédent.
    :return: Booléen.
    """
    return d[key_now]["to"][0] == d[key_before]["to"][-1]


def check_sw_now_equals_to_before(d, key_now, key_before):
    """
    Vérifie si la dernière fréquence est égale à la première du bloc suivant.
    :param d: Container
    :param key_now: Clé de l'objet en cours.
    :param key_before: Clé de l'objet précédent.
    :return: Booléen.
    """
    return d[key_now]["sw"][0] == d[key_before]["to"][-1]


def check_sw_equals_to(d, key_now, key_before):
    """
    Vérifie si la dernière fréquence est égale à la première du bloc suivant.
    :param d: Container
    :param key_now: Clé de l'objet en cours.
    :param key_before: Clé de l'objet précédent.
    :return: Booléen.
    """
    if check_to_now_equals_to_before(d, key_now, key_before) and check_sw_now_equals_sw_before(d, key_now, key_before):
        return d[key_now]["sw"][0] == d[key_now]["to"][0]


def check_sw_now_equals_sw_before(d, key_now, key_before):
    """
    Vérifie si la dernière fréquence est égale à la première du bloc suivant.
    :param d: Container
    :param key_now: Clé de l'objet en cours.
    :param key_before: Clé de l'objet précédent.
    :return: Booléen.
    """
    return d[key_now]["sw"][0] == d[key_before]["sw"][-1]


def check_to_sw_first_equal(d, tag):
    """

    :param d: Container.
    :param tag: Clé
    :return:
    """
    return d[tag]["to"][0] == d[tag]["sw"][0]


def check_to_sw_last_equal(d, tag):
    return d[tag]["to"][-1] == d[tag]["sw"][-1]


def contains_mock(tag):
    return re.search(r"mock", tag) is not None


def contains_pause(tag):
    return re.search(r"pause", tag) is not None


def contains_warmup(tag):
    return re.search(r"warmup", tag) is not None


def contains_tracking(tag):
    return re.search(r"tracking", tag) is not None or re.search(r"warmdown", tag) is not None


def check_ok(d):
    return d


def clean_switches(d, tag):
    first_ok = check_to_sw_first_equal(d, tag)
    last_ok = check_to_sw_last_equal(d, tag)
    size_wrong = check_size(d, tag) < 0

    if last_ok and first_ok and size_wrong:
        block = d[tag]
        # Lequel est le plus long ?
        u_t, c_t = np.unique(block["to"], return_counts=True)
        u_s, c_s = np.unique(block["sw"], return_counts=True)

        if np.array_equal(u_t, u_s):
            diff = c_t - c_s
            # Je récupère la valeur qui est en trop.
            idx = np.where(diff == -1)[0]
            values = u_s[idx]
            to, sw = block["to"], block["sw"]
            # 1er essai.
            delta = get_delta(d, tag)
            to, sw = block["to"], block["sw"]
            if check_small_error(to, sw[delta:]):
                d = chop_front(d, tag, cut=delta, what="sw")
                return d

            elif check_small_error(to, sw[:-delta]):
                d = chop_back(d, tag, cut=-delta, what="sw")
                return d

            for i in range(to.size + 1):
                array_equal = np.array_equal(to[:i], sw[:i])
                if not array_equal:
                    sw = np.hstack((sw[:i-1], sw[i:]))

            for j, value in enumerate(values):
                now = value
                # Prendre les indices des autres valeurs de la liste et retirer ces valeurs.
                to, sw = block["to"], block["sw"]
                array_equal = False
                while not array_equal:
                    pass

    return d


def clean_block(d, tag):
    first_ok = check_to_sw_first_equal(d, tag)
    last_ok = check_to_sw_last_equal(d, tag)
    size_wrong = check_size(d, tag) < 0

    if last_ok and not first_ok and size_wrong:
        delta = get_delta(d, tag)
        d = chop_front(d, tag, cut=delta, what="sw")
        if _check_small_error(d, tag):
            d = set_ok(d, tag)

    elif not last_ok and first_ok and size_wrong:
        # delta = get_delta(d, tag)
        d = chop_back(d, tag, what="sw")
        if _check_small_error(d, tag):
            d = set_ok(d, tag)

    elif last_ok and first_ok and size_wrong:
        d = clean_switches(d, tag)

    elif last_ok and first_ok and not size_wrong:
        if _check_small_error(d, tag):
            d = set_ok(d, tag)

    return d


def clean(d, tag, before):
    """

    :param d:
    :param tag:
    :param before:
    :return:
    """
    # d = clean_switches(d, tag)

    sav_sap = check_sw_now_equals_sw_before(d, tag, before)
    sav_tap = check_to_now_equals_sw_before(d, tag, before)
    tav_tap = check_to_now_equals_to_before(d, tag, before)
    tav_sap = check_sw_now_equals_to_before(d, tag, before)
    t_s_eq = check_sw_equals_to(d, tag, before)
    # Cas sav_sap && tav_tap : je dois chop_front dans le tracking.
    # Cas d'une fréquence poussée.
    if tav_tap:
        if not t_s_eq:
            if not check_to_sw_last_equal(d, before):
                can_chop_back = test_chop_back(d, before, what="sw")
                if can_chop_back:
                    d = chop_back(d, before, what="sw")
                can_chop_front = test_chop_front(d, tag, what="sw")
                if can_chop_front:
                    d = chop_front(d, tag, what="sw")
        else:
            if sav_sap:
                d = chop_front(d, tag, what="sw")  # Suprimer le premier switch.
                d = chop_front(d, tag, what="to")  # Supprime la fréquence poussée.
            else:
                d = chop_front(d, tag, what="to")

    else:
        # Pas triggée dans le mock.
        if sav_tap and sav_sap:
            d = chop_back(d, before, what="sw")  # Supprime le dernier switch.
            d = chop_front(d, tag, what="sw")
            d = chop_front(d, tag, what="to")

        # Pas triggée dans le mock.
        elif sav_tap and not sav_sap:
            # Cas où la fréquence est poussée, mais que l'animal a changé de position et de bloc.
            d = chop_back(d, before, what="sw")  # Supprime le dernier switch.
            d = chop_front(d, tag, what="to")

        elif not sav_tap and sav_sap:
            # d = chop_front(d, tag, what="to")
            d = chop_front(d, tag, what="sw")

        else:  # not sav_tap and not sav_sap
            pass

    if not d[tag]["ok"]:
        d = clean_block(d, tag)

    return d


def check_warmup(d, tag, future, before, mapping):
    # tag = "warmup0"
    wp = d[tag]
    t0 = d[future]
    pause = d[before]

    sav_sap = check_sw_now_equals_sw_before(d, tag, before)
    sav_tap = check_to_now_equals_sw_before(d, tag, before)

    if sav_sap and sav_tap:
        d = chop_front(d, tag, what="sw")
        d = chop_front(d, tag, what="to")

    elif sav_sap and not sav_tap:
        d = chop_front(d, tag, what="sw")

    elif not sav_sap and sav_tap:
        d = chop_front(d, tag, what="to")

    else:
        if check_equal(d, tag):
            d = set_ok(d, tag)
    # d = clean_switches(d, tag)
    d = clean(d, future, tag)
    d = clean_block(d, tag)

    return d


def check_tracking(d, tag, before, playbacks, future=None):
    """
    C'est le bordel.
    :param d:
    :param tag:
    :param before:
    :param playbacks:
    :param future:
    :return:
    """
    # C'est un comportement normal quand deux blocks de tracking se suivent (w0, t0).
    # Aussi quand dernière fréquence de playback == dernière fréquence de mock.
    # Dans ce cas, on supprime la mention de l'indice dans now["sw"] et now["idx"]

    if contains_mock(before):
        d = clean(d, tag, before)

        # La dernière fréquence de playback est égale à fréquence mock, la fréquence n'est pas
        # poussée en sauvegarde dans le tracking. Surtout sw_mock égal à sw_tracking. Donc, on coupe sw.

        # Pb et Mck différents. Donc, on pousse la dernière fréquence de Mock.
    return d


def check_mock(d, tag, before):
    # Avoir le playback pour certain cas ?
    # Why not ?
    d = clean_switches(d, tag)
    d = clean(d, tag, before)
    return d


def check_equal(d, tag):
    return np.array_equal(d[tag]["sw"], d[tag]["to"])


def set_ok(d, tag):
    d[tag]["ok"] = True
    return d


def detect_frequency_switch_2(vec, mapping, mask=None):
    """
    Fonction qui a pour objectif de détecter les changements de fréquences.
    :param mapping:
    :param vec:
    :param mask:
    :return:
    """
    # Avec ça je peux comparer la dernière fréquence d'un bloc, et la comparer au bloc suivant.
    # Je convertis le vecteur de positions en fréquences.
    tone_vec = mapping.convert_to_frequency(vec)
    # Je repère les moments où les fréquences changent.
    idx = np.where(np.diff(np.concatenate(([0], tone_vec))))[0]
    if mask is not None:
        assert (mask.size == vec.size)
        in_mask = mask[idx]
        idx = idx[in_mask]
    # Je retourne un array de fréquences.
    switch = tone_vec[idx]
    return switch, idx


def find_cut(t, s):
    """

    :param t: Tones
    :param s: Switches
    :return:
    """
    delta = t.size - s.size
    assert(t[delta:].size == s.size)
    if np.array_equal(t[delta:], s):
        return delta
    elif np.array_equal(t[:-delta], s):
        return -delta
    else:
        if check_small_error(t[delta:], s):
            return delta

        if check_small_error(t[:-delta], s):
            return -delta

        else:
            return 0


def foo(folder):
    delay_frame = 0.007  # secondes
    delay_frame_samples = int(delay_frame * 30e3)
    mapping = Mapping(1920, 33, 2000., 7)
    # Charger le tt.npz
    sequence = gd.extract_data(folder)
    # Charger les positions
    files, types, nums = gather_positions_file(folder, sequence.get_n_iter())
    # hstack positions
    positions, separators = load_positions_file(files)
    # clean_positions
    positions = clean_positions(positions)
    # Couper selon les séparations.
    positions = split(positions, separators)

    # L'idée maintenant, c'est d'itérer sur l'objet séquence.
    # Alternance Tracking / Playback
    triggers = list()
    test_triggers = list()
    separators = list()
    d = list()
    for i, triplet in enumerate(zip(types, nums, positions)):
        t, num, pos = triplet
        if t == "pause":
            continue

        if t == "playback":
            t = "mock"

        block = sequence.get_xp_number(t, num)
        switches, idx = detect_frequency_switch(pos, mapping)
        # idx : indice du changement de fréquence dans les positions.
        # À peu près tous les blocks ont plus de fréquences que ce que vont dire les positions.
        # print(switches.size - block.tones.size)
        # Ici : il manque un switch dans ce qui est calculé à partir des positions.
        if len(block.tones) == len(switches) + 1:
            if np.array_equal(switches, block.tones[1:]):
                past_positions = positions[i - 1]
                past_switches, _ = detect_frequency_switch(past_positions, mapping)
                last_tone = past_switches[-1]
                if last_tone != block.tones[0]:
                    last_tone = block.tones[0]
                switches = np.hstack((np.array(last_tone), switches))
                idx = np.hstack((np.array(0, dtype=int), idx))
            elif np.array_equal(switches, block.tones[:-1]):
                future_positions = positions[i + 1]
                future_switches, _ = detect_frequency_switch(future_positions, mapping)
                next_tone = future_switches[0]
                switches = np.hstack((switches, np.array(next_tone)))
                # idx =

        elif len(switches) == len(block.tones) + 1:
            if np.array_equal(switches[1:], block.tones):
                print("Debut S == Tones")
            elif np.array_equal(switches[:-1], block.tones):
                print("Fin S == Tones")
            switches = switches[:-1]
            idx = idx[:-1]
        assert(np.array_equal(switches, block.tones))
        # Récupérer les triggers.
        # sound_triggers = block.triggers
        p_triggers = np.zeros_like(pos)
        seps = np.zeros_like(pos)
        seps[0] = 1
        for j, trigger in enumerate(idx):
            p_triggers[trigger] = block.triggers[j]
        triggers.append(p_triggers)
        # indices = np.where(p_triggers != 0)[0]
        # for i0, i1 in zip(indices[:-1], indices[1:]):
        #     segment = p_triggers[i0:i1]
        #     t0, t1 = p_triggers[i0], p_triggers[i1]
        #     if len(segment) > 1:
        #         dt = int((t1 - t0) / len(segment))
        #         if dt < 100:
        #             print(dt)
        #         p_triggers[i0:i1] = np.full_like(segment, t0) + np.arange(len(segment), dtype=int) * dt

        _d = {"Type": t, "Number": num, "Positions": pos, "Triggers": p_triggers}
        # triggers.append(p_triggers)
        separators.append(seps)
        d.append(_d)

    # Enlever les zéros contigus, récupérer les indices.
    # Si le début est nul.
    triggers = clean_triggers(np.hstack(triggers))
    separators = np.hstack(separators)
    triggers -= delay_frame_samples

    u, c = np.unique(np.diff(triggers), return_counts=True)
    print(u[u < 100], c[u < 100].size)
    plt.plot(triggers)
    plt.show()

    triggers = split(triggers, separators)

    for i, elt in enumerate(triggers):
        d[i]["Triggers"] = elt

    # à partir de là, je peux créer un objet ttp.
    pairs = list()
    for i, _d in enumerate(d):
        t, n, p, m = _d["Type"], _d["Number"], _d["Positions"], _d["Triggers"]
        pairs.append(PositionPair(p, m, t, number=n, order=i))

    spt = SequencePT(n_iter=sequence.get_n_iter(), folder=None)
    for pair in pairs:
        spt.add(pair)

    spt.save(folder)


def check_small_error(t, s):
    array_equal = np.array_equal(t, s)
    equal_length = t.size == s.size
    if equal_length and not array_equal:
        idx = np.equal(t - s, 0)
        if sum(~idx) < 2:
            array_equal = True
    return array_equal


def _check_small_error(d, tag):
    t, s = d[tag]["to"], d[tag]["sw"]
    return check_small_error(t, s)


def mask_before(x, tone, mapping):
    """

    :param x: Positions.
    :param tone: Fréquence à rechercher.
    :param mapping:
    :return:
    """
    mask = np.full_like(x, True, dtype=bool)
    s, idx = detect_frequency_switch(x, mapping)
    if s[0] == tone:
        mask[:idx[1]] = False
    return mask


def fill_positions_triggers(d):
    """

    :param d:
    :return:
    """
    triggers = list()
    separators = list()
    mask = list()
    min_dt = 750
    mean_dt = 1000
    for key, value in d.items():
        if contains_pause(key):
            continue
        p = value["pos"]
        btr = value["tr"]
        idx = value["idx"]
        p_triggers = np.zeros_like(p)
        seps = np.zeros_like(p_triggers)
        m = value["m"]
        seps[0] = 1
        di = np.diff(idx)
        dt = np.diff(btr)

        for i, triplet in enumerate(zip(idx[1:], di, dt)):
            _idx, _di, _dt = triplet
            samples = _di * min_dt
            if samples > _dt:
                # TODO : calculer l'indice.
                jump = np.ceil(_dt / mean_dt)
                idx[i] = _idx + jump

        for j, trigger in enumerate(idx):
            p_triggers[trigger] = btr[j]

        triggers.append(p_triggers)
        separators.append(seps)
        mask.append(m)

    return np.hstack(triggers), np.hstack(separators)


def fill_positions_triggers_2(d):
    """

    :param d:
    :return:
    """
    triggers = list()
    idx = list()
    separators = list()
    pt = list()
    min_dt = 500
    mean_dt = 1000
    size = 0

    for key, value in d.items():
        if contains_pause(key):
            continue
        p = value["pos"]
        triggers.append(value["tr"])
        idx.append(value["idx"] + size)
        size += p.size
        pt.append(np.zeros_like(p))
        seps = np.zeros_like(p)
        seps[0] = 1
        separators.append(seps)

    pt = np.hstack(pt)
    triggers = np.hstack(triggers)
    idx = np.hstack(idx)
    di = np.diff(idx)  # np.concatenate(([0], np.diff(idx)))
    dt = np.diff(triggers)  # np.concatenate(([0], np.diff(triggers)))
    separators = np.hstack(separators)
    where = di * min_dt > dt
    jumps = np.ceil(dt[where] / mean_dt)
    where = np.concatenate(([False], where))
    idx[where] = idx[where] + jumps

    # index 232546 is out of bounds for axis 0 with size 224985 -> 800

    for i, trigger in enumerate(idx):
        pt[trigger] = triggers[i]

    # plt.plot()
    # for i, triplet in enumerate(zip(idx[1:], di, dt)):
    #     _idx, _di, _dt = triplet
    #     samples = _di * min_dt
    #     if samples > _dt:
    #         # TODO : calculer l'indice.
    #         jump = np.ceil(_dt / mean_dt)
    #         idx[i] = _idx + jump

    return pt, separators


def chop_front(d, tag, cut=1, what="sw"):
    assert(what in ["sw", "to"])
    block = d[tag]
    if what == "sw":
        block["sw"] = block["sw"][cut:]
        block["idx"] = block["idx"][cut:]

    else:
        block["to"] = block["to"][cut:]
        block["tr"] = block["tr"][cut:]

    block["ok"] = check_small_error(block["to"], block["sw"])
    d[tag] = block
    return d


def chop_back(d, tag, cut=-1, what="sw"):
    assert(what in ["sw", "to"])
    block = d[tag]
    if what == "sw":
        block["sw"] = block["sw"][:cut]
        block["idx"] = block["idx"][:cut]

    else:
        block["to"] = block["to"][:cut]
        block["tr"] = block["tr"][:cut]

    block["ok"] = np.array_equal(block["sw"], block["to"])
    d[tag] = block
    return d


def test_chop_front(d, tag, cut=1, what="sw"):
    assert (what in ["sw", "to"])
    block = d[tag]
    if what == "sw":
        return block["sw"][1] == block["to"][0]

    else:
        return block["sw"][0] == block["to"][1]


def test_chop_back(d, tag, cut=-1, what="sw"):
    assert (what in ["sw", "to"])
    block = d[tag]
    if what == "sw":
        return block["sw"][-2] == block["to"][-1]

    else:
        return block["sw"][-1] == block["to"][-2]



def feel_lucky(d, tag, before):
    if contains_warmup(tag):
        return d
    swp_tof = check_to_now_equals_sw_before(d, tag, before)
    size_ok = check_size(d, tag)
    sw_eq_to = check_to_sw_first_equal(d, tag)
    l_sw_eq_to = check_to_sw_last_equal(d, tag)
    if size_ok == 0 and sw_eq_to and l_sw_eq_to and not swp_tof:
        d = set_ok(d, tag)
    return d


def get_delta(d, tag):
    block = d[tag]
    delta = block["sw"].size - block["to"].size
    return delta


def debugging(folder):
    delay_frame = 0.007  # secondes
    delay_frame_samples = int(delay_frame * 30e3)
    mapping = Mapping(1920, 33, 2000., 7)
    # Charger le tt.npz
    sequence = gd.extract_data(folder)
    # Charger les positions
    files, types, nums = gather_positions_file(folder, sequence.get_n_iter())
    # hstack positions
    positions, separators, spam = load_positions_file(files)
    # clean_positions
    positions = clean_positions(positions)

    sw_total, idx_total = detect_frequency_switch(positions, mapping)
    spam_cp = spam[idx_total]
    # Compter le nombre de 1 avant.
    # np.where(np.diff(spam) == 0)
    # sw_total = split(sw_total, separators)
    # Couper selon les séparations.
    positions = split(positions, separators)

    ########
    # TEST #
    ########
    n_blocks = sequence.get_n_iter()
    tags_for_order = list()
    is_ok = dict()
    playbacks = dict()
    for i, triplet in enumerate(zip(types, nums, positions)):
        t, num, pos = triplet
        # if t not in ["warmup", "pause" "tracking"] and num != 0:
        #     continue
        # if t == "warmdown":
        #     continue
        if t == "playback":
            t = "mock"
            playbacks[t + str(num)] = (sequence.get_xp_number("playback", num))
        tag = t + str(num)
        btr, bto, array_equal = None, None, False
        sw, indices = detect_frequency_switch_2(pos, mapping)
        m = np.full_like(pos, False, dtype=bool)
        tags_for_order.append(tag)
        if t != "pause":
            block = sequence.get_xp_number(t, num)
            btr = deepcopy(block.triggers)
            bto = deepcopy(block.tones)
            # sw, indices = detect_frequency_switch(pos, mapping)
            _spam_idx = np.where(spam_cp == i)[0]

            m = np.full_like(pos, True, dtype=bool)
        is_ok[tag] = {"ok": array_equal, "pos": pos, "tr": btr,
                      "to": bto, "sw": sw, "m": m, "idx": indices, "what": t}
    before = tags_for_order[0]
    triggers = list()
    for key, value in is_ok.items():
        if value["what"] == "pause":
            continue
        p_triggers = np.zeros_like(value["pos"])

    for i, tag in enumerate(tags_for_order):
        now = is_ok[tag]
        future = None
        if i + 1 < len(tags_for_order):
            future = tags_for_order[i + 1]
        if now["what"] == "pause":
            before = tag
            continue

        is_ok = feel_lucky(is_ok, tag, before)

        if contains_warmup(tag):
            is_ok = check_warmup(is_ok, tag, future, before, mapping)

        elif contains_tracking(tag):
            is_ok = check_tracking(is_ok, tag, before, playbacks, future)

        elif contains_mock(tag):
            is_ok = check_mock(is_ok, tag, before)

        before = tag

    score = 0
    total = 0
    for key, value in is_ok.items():
        if contains_pause(key):
            continue
        score += int(value["ok"])
        total += 1
    print(f"{score} / {total}")

    for key, value in is_ok.items():
        # print(key)
        if contains_pause(key):
            continue
        # if not value["ok"]:
        #     print(key)
    if score < total:
        failed = dict()
        for key, value in is_ok.items():
            if not value["ok"] and value["what"] != "pause":
                failed[key] = value
        print(failed.keys())
        print("DEBUG")
    else:
        triggers, separators = fill_positions_triggers_2(is_ok)
        triggers = clean_triggers(triggers)
        triggers -= delay_frame_samples
        plt.plot(triggers)
        plt.show()
        triggers = split(triggers, separators)
        for trigger in triggers:
            u, c = np.unique(np.diff(trigger), return_counts=True)
            print(sum(c[np.less(u, 100)]))
    # for trigger in triggers:
    #     u, c = np.unique(np.diff(trigger), return_counts=True)
    #     print(sum(np.less(u, 100)))

    #


if __name__ == "__main__":
    fishy = ["/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230219/MUROLS_20230219_SESSION_00",
             "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230218/MUROLS_20230218_SESSION_01",
             "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230221/MUROLS_20230221_SESSION_00",
             "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230220/MUROLS_20230220_SESSION_00",
             ]
    from tqdm import tqdm
    mock_problems = ["/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230224/MUROLS_20230224_SESSION_00",
                     "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230227/MUROLS_20230227_SESSION_00",
                     "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230301/MUROLS_20230301_SESSION_00"
                     ]
    last_mock_problems = ["/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230301/MUROLS_20230301_SESSION_00"]
    folders, save_folder = get_good_sessions("MUROLS")
    # TODO : gérer les cas où les tones sont vides.
    folders.sort()
    folders = folders[::-1]
    for folder in folders:
        # print(folder)
        debugging(folder)
    # ICI : regarder si la pause 1 est pleine de -1. Non.
    minus_one_list = ["/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230220/MUROLS_20230220_SESSION_00",
                      "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230221/MUROLS_20230221_SESSION_00",
                      "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230218/MUROLS_20230218_SESSION_01",
                      "/Users/flavienferal/data/EXPERIMENT/MUROLS/MUROLS_20230219/MUROLS_20230219_SESSION_00"]

    # for folder in minus_one_list:
    #     debugging(folder)
