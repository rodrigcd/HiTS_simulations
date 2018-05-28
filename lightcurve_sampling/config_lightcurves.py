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
sn_lightcurves_path = "/home/rcarrasco/simulated_data/pickles/hits_lightcurves_100000.pkl"
sn_parameters_path = "/home/rcarrasco/simulated_data/pickles/hits_params_100000.pkl"
save_path = "/home/rcarrasco/simulated_data/lightcurves/"

# Configuration Variables
requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
                        "NonVariable", "EmptyLightCurve", "Asteroids"]
# requested_lightcurve = ["EclipsingBinaries"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5, 6]  # multiclass
# requested_lightcurve_labels = [3, ]
file_name = "custom_distr"
bands = ["g", ]
n_per_class_per_field = 200
n_std_limmag = 0.7  # How deep you want to sample the magnitudes for sampling
file_name = file_name + str(n_per_class_per_field)

magnitude_upper_limit = 14.5
magnitude_lower_limit = {"RRLyrae": 22.7, "M33Cepheids": 23.5}

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
M33_cepheids_path = "./lc_data/cepheid_gps.pkl"
eb_path = "./lc_data/eb_templates.pkl"
