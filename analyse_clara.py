from kneed import DataGenerator, KneeLocator
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import scipy.io
import json
import numpy as np
from format_data import *
from utils import *

t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
#gc = np.arange(0, 32)


path = '/mnt/working2/felicie/data2/eTheremin/ALTAI/ALTAI_20240722_SESSION_02/'
type = get_session_type_final(path)


data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)
features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)
#gc = np.load(path+'headstage_0/good_clusters.npy', allow_pickle=True)
gc = np.arange(0, 32)

# afficher les psth par cluster

# afficher la moyenne des psth de tous les clusters