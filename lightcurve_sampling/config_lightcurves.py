# Absolute paths
# Hayabusa
# sn_lightcurves_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves_100000.pkl"
# sn_parameters_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params_100000.pkl"
# save_path = "/home/toshiba/rodrigo/simulated_lightcurves/HiTS_lc/"
# Personal
# sn_lightcurves_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves.pkl"
# sn_parameters_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params.pkl"
# save_path = "/home/rodrigo/supernovae_detection/simulated_data/lightcurves/"
# Cluster
# sn_lightcurves_path = "/home/rcarrasco/simulated_data/from_surveysim/hits_lightcurves_100000.pkl"
# sn_parameters_path = "/home/rcarrasco/simulated_data/from_surveysim/hits_params_100000.pkl"


type1_sn_lc_path = "/home/rcarrasco/surveysim/pickles/Hsiao_hits_20000_lc.pkl"
type1_params_path = "/home/rcarrasco/surveysim/pickles/Hsiao_hits_20000_params.pkl"
type2_sn_lc_path = "/home/rcarrasco/surveysim/pickles/mid_bounded_lowz_20000_lc.pkl"
type2_params_path = "/home/rcarrasco/surveysim/pickles/mid_bounded_lowz_20000_params.pkl"
"""
save_path = "/home/rcarrasco/simulated_data/lightcurves/"

# Configuration Variables
# requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
#requested_lightcurve = ["TypeISupernovae", "TypeIISupernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
#                        "NonVariable", "EmptyLightCurve", "Asteroids"]
# requested_lightcurve = ["EclipsingBinaries"]
#requested_lightcurve_labels = [0, 1, 2, 3, 4, 5, 6, 7]  # multiclass
# requested_lightcurve_labels = [3, ]
requested_lightcurve = ["TypeISupernovae", "TypeIISupernovae"] # To create templates
requested_lightcurve_labels = [0, 1] # To create templates
file_name = "templates"

bands = ["g", "r", "i"]
n_per_class_per_field = 5000
n_std_limmag = 0.7  # How deep you want to sample the magnitudes
file_name = file_name + str(n_per_class_per_field)

magnitude_upper_limit = 14.5
#magnitude_lower_limit = {"RRLyrae": 22.7, "M33Cepheids": 23.5}
magnitude_lower_limit = {}
standard_erf_limit = {"g": 22.8}
custom_erf_limit = {"RRLyrae": {"g": 21.5}}

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
M33_cepheids_path = "./lc_data/cepheid_gps.pkl"
eb_path = "./lc_data/eb_templates.pkl"
"""
## ZTF SIMULATIONS

save_path = "/home/rcarrasco/simulated_data/lightcurves/"
requested_lightcurve = ["RRLyrae", "M33Cepheids", "EclipsingBinaries", "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5]
file_name = "ztf_no_sn"
bands = ["g", "r"]

n_per_class_per_field = 10
n_std_limmag = 0.7  # How deep you want to sample the magnitudes
file_name = file_name + str(n_per_class_per_field)

magnitude_upper_limit = 10
#magnitude_lower_limit = {"RRLyrae": 22.7, "M33Cepheids": 23.5}
magnitude_lower_limit = {}
standard_erf_limit = {"g": 20.5, "r":20-0.5}
custom_erf_limit = {}#{"RRLyrae": {"g": 21.5}}

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/ztf_conditions.pkl"
M33_cepheids_path = "./lc_data/cepheid_gps.pkl"
eb_path = "./lc_data/eb_templates.pkl"


