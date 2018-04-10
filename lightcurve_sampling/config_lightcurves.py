import numpy as np

""" -------------------- Data Paths ---------------------"""
# To days and limmag per band
observation_data = np.load(
    "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_days_limmag_sky500000.pkl")
#observation_data = np.load("/home/rodrigo/surveysim/pickles/MoriyaWindAcc_SNLS_bands-nf1-ne1-nr5-nn8855_CFHT-MegaCam_girz_days_limmag_sky1000.pkl")



# supernovae light curves and parameters path (by Francisco's simulations)
lc_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_LCs_500000.pkl"
param_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_params_500000.pkl"
#lc_path = "/home/rodrigo/surveysim/pickles/MoriyaWindAcc_SNLS_bands-nf1-ne1-nr5-nn8855_CFHT-MegaCam_girz_LCs_1000.pkl"
#param_path = "/home/rodrigo/surveysim/pickles/MoriyaWindAcc_SNLS_bands-nf1-ne1-nr5-nn8855_CFHT-MegaCam_girz_params_1000.pkl"

# Magnitude distribution path (by Jorge Martinez)
distr_path = "/home/rodrigo/supernovae_detection/object_distribution/"
#distr_path = "/home/rodrigo/supernova_detection/object_distribution/"

# Cepheids light curve paths by LSST simulator
cepheids_path = "/home/toshiba/rodrigo/lsst_data/pablo_simulation/CEPH_SNLS_compress.pkl"
M33_cepheids_path = "/home/toshiba/rodrigo/cepheids_lc/auto_selected_gp_v2.pkl"
#cepheids_path = "/home/rodrigo/supernova_detection/cepheids_lc/CEPH_SNLS_compress.pkl"

# Save path
save_path = "/home/toshiba/rodrigo/simulated_lightcurves/"
#save_path = "/home/rodrigo/supernova_detection/simulated_lightcurves/"

# File name to save
file_name = "multiclass_SNLS_15_12_"

""" ----------------- Data Generation Parameters ----------------"""
# Requested light curves per type and label as int
requested_lightcurve = ["Supernovae", "RRLyrae", "Constant", "EmptyLightCurve", "M33Cepheids", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5]  # multiclass
#requested_lightcurve_labels = [0, 1, 1, 1, 1]
#requested_lightcurve_labels = [0, 0, 1, 1, 0]

# Number of light curves per class
n_per_class = 100000
file_name += str(n_per_class)

# Bands to produce light curves
bands = ["g", "r", "i"]

# Build dictionaries with observations days and limmag
observation_days = {}
extrapolation_limits = {}
band_ex = "r"
limmag = {}
for band in bands:
    extrapolation_limits[band] = [12, np.mean(observation_data[band]["limmag"])]
    observation_days[band] = observation_data[band]["days"]
    limmag[band] = observation_data[band]["limmag"]
extend_fit = 0

# Fit limits for extrapolation of empirical distributions
fit_limits = {"z": [17.5, 20 + extend_fit],
              "u": [16, 21 + extend_fit],
              "g": [19, 22 + extend_fit],
              "r": [18, 22 + extend_fit],
              "i": [18, 20 + extend_fit]}

print("----- SNLS Lightcurves -----")