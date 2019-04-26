import os
import sys

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PATH_TO_PROJECT)

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from image_simulation.config_images import *
import matplotlib.gridspec as gridspec
import pickle as pkl



if __name__ == '__main__':
  save_path = "/home/ereyes/Projects/Alerce/AlerceDHtest/datasets/ZTF/simulated_data/ztf_positive_psf10.pkl"
  f = np.load(save_path)

