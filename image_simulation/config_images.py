
# Absolute paths
# Hayabusa
# save_path = "/home/toshiba/rodrigo/simulated_stamps/HiTS_stamps/"
# galaxy_path = "/home/rodrigo/supernovae_detection/galaxies_guille/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
# lightcurves_path = "/home/toshiba/rodrigo/simulated_lightcurves/HiTS_lc/hits_2500"
# Personal
#save_path = "/simulated_data/" # System directory with more space
#save_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/"
#galaxy_path = "/home/rodrigo/supernovae_detection/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
#lightcurves_path = "/home/rodrigo/supernovae_detection/simulated_data/lightcurves/hits_30"
# Cluster
"""
save_path = "/home/rcarrasco/simulated_data/image_sequences/"
galaxy_path = "/home/rcarrasco/simulated_data/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
lightcurves_path = "/home/rcarrasco/simulated_data/lightcurves/"
lightcurve_name = "eb_more_detections2500"
lightcurves_path = lightcurves_path + lightcurve_name

# Configuration Variables
# requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "EclipsingBinaries",
#                        "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve = ["EclipsingBinaries"]
# requested_lightcurve_labels = [0, 1, 2, 3, 4, 5, 6]  # multiclass
requested_lightcurve_labels = [3,]
bands = ["g", ]
stamp_size = (21, 21)
proportion_with_galaxy = [0.5, 0.01, 0.01, 0.01, 0.01, 1.0, 0]
lc_per_chunk = 2000
sky_clipping = 2200
augmented_psfs = False
astrometric_error = 0.3
output_filename = "eb_images"
output_filename = output_filename + "_" + lightcurve_name

filter_by_conditions = {"seeing": {"g": [0, 2.0 / 0.27]}}
                        #"zero_point": {"g": [24.9, 25.15]}}

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
"""
# ################### ZTF SIMULATION ###############################

save_path = "/home/rcarrasco/simulated_data/image_sequences/"
galaxy_path = "/home/rcarrasco/simulated_data/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
lightcurves_path = "/home/rcarrasco/simulated_data/lightcurves/"
lightcurve_name = "good_zero_points10"
lightcurves_path = lightcurves_path + lightcurve_name

requested_lightcurve = ["RRLyrae", "M33Cepheids", "EclipsingBinaries", "NonVariable", "EmptyLightCurve", "Asteroids"]
requested_lightcurve_labels = [0, 1, 2, 3, 4, 5]


bands = ["g", "r"]
stamp_size = (21, 21)
proportion_with_galaxy = [0, 0, 0, 0, 1.0, 0]
lc_per_chunk = 50
sky_clipping = 2200
augmented_psfs = False
astrometric_error = 0.3
output_filename = "good_zero_points"
output_filename = output_filename + "_" + lightcurve_name

filter_by_conditions = {}
                        #"zero_point": {"g": [24.9, 25.15]}}

# Don't worry about this
camera_and_obs_cond_path = "../real_obs/pickles/ztf_conditions_postive_psfs_v5.pkl"
