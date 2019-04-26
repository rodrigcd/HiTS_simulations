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

estimated_counts_key = 'estimated_counts'
estimated_error_counts_key = 'estimated_error_counts'
g_key = 'g'
r_key = 'r'
lightcurves_key = 'lightcurves'

"""
ADU == estimated counts == flux == f
zp: zero point (24,5)
T: exposure time (30s)
m: magnitude
"""


def get_magnitude(ADU, zp, T):
  magnitude = zp - 2.5 * np.log10(ADU / T)
  #if np.isnan(magnitude).any():
  #  print("ADU %s\nT %s\nADU/T %s\nlog %s"
  #        % (str(ADU), str(T), str(ADU / T), str(np.log10(ADU / T))))
  return magnitude


"""
How to calculate this can be seen in 
http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html

Thus 
ADU = flux = f
magnitude_error = sigma_m; ADU(flux)_error = sigma_f 
sigma_m = 1.09*(sigma_f/f)

Estimated error count = Var of ADU = sigma_f**2
"""


def get_magnitude_error(estimated_counts, estimated_count_error):
  f = estimated_counts
  sigma_f = np.sqrt(estimated_count_error)
  sigma_m = 1.09 * (sigma_f / f)
  return sigma_m


def _silent_plot_underliying(ax, field_data, lc_idx, plt_marker='o-'):
  field_lc = field_data[lightcurves_key]
  days = field_data["obs_cond"]["obs_day"]
  bands = list(field_data[lightcurves_key].keys())
  for band in bands:
    ax.plot(days[band], field_lc[band][lc_idx], plt_marker, label=band)
  ax.legend()
  ax.set_ylim([21, 15])
  galaxy_flag = field_data["galaxy_flag"][lc_idx]
  lc_type = field_data["lc_type"][lc_idx]
  ax.set_title("%s; galaxy_flag: %i" % (lc_type, galaxy_flag), fontsize=15)
  return ax


def _silent_plot_lc(ax, field_data, lc_idx, plt_marker='o-'):
  field_lc = field_data[lightcurves_key]
  estimated_counts = field_data[estimated_counts_key]
  estimated_error_counts = field_data[estimated_error_counts_key]
  days = field_data["obs_cond"]["obs_day"]
  bands = list(field_data[lightcurves_key].keys())
  for i, band in enumerate(bands):
    magnitude_noisy = get_magnitude(estimated_counts[band][lc_idx], 24.5, 30)
    magnitude_error = get_magnitude_error(estimated_counts[band][lc_idx],
                                          estimated_error_counts[band][lc_idx])
    # print(estimated_counts[band][lc_idx].shape)
    # print(magnitude_error)
    # print(days[band])
    # print(magnitude_noisy)
    ax.errorbar(days[band][:], magnitude_noisy, yerr=magnitude_error, fmt='o',
                label='%s; lightcurve' % band)
    color = ax.get_lines()[i * 2].get_color()
    ax.plot(days[band], field_lc[band][lc_idx], plt_marker, label='%s; model' % band,
            color=color)
  ax.legend()
  #ax.set_ylim([21, 15])
  galaxy_flag = field_data["galaxy_flag"][lc_idx]
  lc_type = field_data["lc_type"][lc_idx]
  ax.set_title("%s; galaxy_flag: %i" % (lc_type, galaxy_flag), fontsize=15)
  return ax


def _get_field(data, field):
  if field is None:
    fields = list(data.keys())
    field = fields[int(np.random.choice(np.arange(len(fields)), 1, replace=False))]
    return field
  else:
    return field


def plot_underliying_model(data, field=None, n_to_plot=1):
  field = _get_field(data, field)
  field_data = data[field]
  lc_idxs_to_plot = np.random.choice(
      np.arange(field_data[lightcurves_key][g_key].shape[0]), n_to_plot, replace=False)
  for lc_idx in lc_idxs_to_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    _silent_plot_underliying(ax, field_data, lc_idx)
    plt.show()


def plot_underliying_and_lc_model(data, field=None, n_to_plot=1):
  field = _get_field(data, field)
  field_data = data[field]
  lc_idxs_to_plot = np.random.choice(np.arange(field_data[lightcurves_key][g_key].shape[0]), n_to_plot, replace=False) #[13]
  print('%s%s' % (field, str(lc_idxs_to_plot)))
  for lc_idx in lc_idxs_to_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    _silent_plot_lc(ax, field_data, lc_idx)
    plt.show()


if __name__ == '__main__':
  # save_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/"
  save_path = "/home/ereyes/Projects/Alerce/AlerceDHtest/datasets/ZTF/simulated_data/image_sequences/ztf_positive_psf_ztf_positive_psf10"
  f = h5py.File(save_path + ".hdf5", "r")
  # stats = np.load("/home/toshiba/rodrigo/simulated_lightcurves/multiclass_SNLS_short20000.pkl")
  exp_time = 30
  zp = 24.5
  fields = list(f.keys())

  channels = [g_key, r_key]
  counts = f["Field01"]['estimated_counts']['g'][0]
  #check field04 y [13
  plot_underliying_and_lc_model(f, n_to_plot=10)#, field='Field04')
