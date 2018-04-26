
# Absolute paths
# Hayabusa
# save_path = "/home/toshiba/rodrigo/simulated_stamps/HiTS_stamps/"
# galaxy_path = "/home/rodrigo/supernovae_detection/galaxies_guille/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
# lightcurves_path = "/home/toshiba/rodrigo/simulated_lightcurves/HiTS_lc/hits_50"
# Personal
save_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/"
galaxy_path = "/home/rodrigo/supernovae_detection/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
lightcurves_path = "/home/rodrigo/supernovae_detection/simulated_data/lightcurves/hits_50"

# Configuration Variables
requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
                        "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5, 6]  # multiclass
bands = ["g", ]
stamp_size = (21, 21)
proportion_with_galaxy = [0.5, 0.05, 0.05, 0.05, 0.05, 0.5, 0]
lc_per_chunk = 120
sky_clipping = 2000
astrometric_error = 0.3
output_filename = "hits_first_try"

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
