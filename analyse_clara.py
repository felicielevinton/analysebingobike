{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from kneed import DataGenerator, KneeLocator\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import csv\n",
    "import os\n",
    "import scipy.io\n",
    "import json\n",
    "import numpy as np\n",
    "from format_data import *\n",
    "from utils import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "t_pre = 0.2#0.2\n",
    "t_post = 0.50#0.300\n",
    "bin_width = 0.005\n",
    "# Créer les bins de temps\"\n",
    "psth_bins = np.arange(-t_pre, t_post, bin_width)\n",
    "gc = np.arange(0, 32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#\n",
    "path = '/mnt/working2/felicie/data2/eTheremin/ALTAI/ALTAI_20240722_SESSION_02/'\n",
    "type = get_session_type_final(path)\n",
    "print(type)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = np.load(path+'headstage_0/data_0.005.npy', allow_pickle=True)\n",
    "features = np.load(path+'headstage_0/features_0.005.npy', allow_pickle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#tracer le psth moyen par cluster (par neurone)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#tracer le psth moyen pour tous les clusters confondus"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
