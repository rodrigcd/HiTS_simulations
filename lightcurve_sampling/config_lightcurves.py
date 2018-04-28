# Absolute paths
# Hayabusa
sn_lightcurves_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves_100000.pkl"
sn_parameters_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params_100000.pkl"
save_path = "/home/toshiba/rodrigo/simulated_lightcurves/HiTS_lc/"
# Personal
# sn_lightcurves_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves.pkl"
# sn_parameters_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params.pkl"
# save_path = "/home/rodrigo/supernovae_detection/simulated_data/lightcurves/"

# Configuration Variables
requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
                        "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5, 6]  # multiclass
file_name = "hits_"
bands = ["g", ]
n_per_class_per_field = 2500
n_std_limmag = 0.7  # How deep you want to sample the magnitudes for sampling
file_name = file_name + str(n_per_class_per_field)

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
M33_cepheids_path = "./lc_data/cepheid_gps.pkl"
eb_path = "./lc_data/eb_templates.pkl"
