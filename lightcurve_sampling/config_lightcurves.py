
# Configuration Variables
requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5]  # multiclass
save_path = "/home/toshiba/rodrigo/simulated_lightcurves/"
file_name = "hits_"
bands = ["g", ]
camera_and_obs_cond_path = "../real_obs/camera_and_obs_cond.pkl"
sn_lightcurves_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves.pkl"
sn_parameters_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params.pkl"
M33_cepheids_path = "./lc_data/cepheid_gps.pkl"
n_per_class_per_field = 1000
n_std_limmag = 1.2
file_name = file_name + str(n_per_class_per_field)
