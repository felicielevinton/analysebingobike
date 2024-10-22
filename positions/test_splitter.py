from motions import find_turns, process_positions
from MotionEntities import Motion
import numpy as np
from resync import Mapping
import plotly.graph_objects as go


if __name__ == "__main__":
    fps = 30
    threshold = 1  # secondes
    n_frames_for_steady = int(fps * threshold)
    data = np.load("FRINAULT_4.npy")
    m, p = process_positions(data)

    l_m = list()
    ma = Mapping(1920, 33, 2000, 7)
    max_length = 0
    whr_max = 0
    for i, segment in enumerate(m):
        if len(segment[1]) > max_length:
            max_length = len(segment[1])
            whr_max = i
        mo = Motion(segment[1], segment[0], 30, ma)
        l_m.append(mo)

    sub_motions = list()
    for i, mot in enumerate(l_m):
        # print(i)
        _xx, _ = find_turns(mot.x, threshold=2, framerate=fps)
        # print(_xx)
        # print(len(_xx))
        for _x in _xx:
            # print(_x)
            # print(len(_x))
            sub_motions.append(Motion(_x, np.arange(len(_x)), 30, ma))

    # print(len(sub_motions))
    l_features = list()
    # l_start_stop = list()
    for mo in sub_motions:
        # l_start_stop.append(np.array(mo.get_start_stop()))
        l_features.append(mo.get_features())

    l_features = np.vstack(l_features)

    fig = go.Figure(data=[
        go.Scatter3d(x=l_features[:, 0], y=l_features[:, 1], z=l_features[:, 2], mode='markers', marker=dict(size=1))
    ])

    fig.update_layout(
        scene=dict(xaxis_title=f"Speed (px.s-1)", yaxis_title=f"Distance (px)", zaxis_title=f"Duration (s)"))

    fig.show()


