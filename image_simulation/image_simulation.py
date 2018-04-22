import numpy as np
from ImageFactory import ImageFactory
import h5py
from MagCounts import Mag2Counts
import time
import sys
import pickle
import matplotlib.pyplot as plt


class ImageDatabase(object):

    def __init__(self, **kwargs):
        # Path
        self.save_path = kwargs["save_path"]
        self.output_filename = kwargs["output_filename"]
        self.bands = kwargs["bands"]
        self.galaxies_distr_path = kwargs["galaxies_distr_path"]
        self.lc_path = kwargs["lc_path"]
        self.camera_and_obs_cond = np.load(kwargs["cam_and_obs_cond_path"])
        self.camera_params = self.camera_and_obs_cond["camera_params"]
        self.obs_cond = self.camera_and_obs_cond["obs_conditions"]
        # Light curves to simulate
        self.requested_lightcurves = kwargs["requested_lightcurves"]
        self.requested_lightcurves_labels = kwargs["requested_lightcurves_labels"]
        self.stamp_size = kwargs["stamp_size"]
        self.prop_lightcurves_with_galaxies = kwargs["prop_lightcurves_with_galaxies"]
        # Instrument and observation conditions
        self.lc_per_chunk = kwargs["lc_per_chunk"]
        self.sky_clipping = kwargs["estimated_sky_clipping"]
        # Light curves parameters
        self.astrometric_error = kwargs["astrometric_error"]
        self.image_stacking_time = kwargs["image_stacking_time"]

        self.image_factory = ImageFactory(nx=self.stamp_size[0],
                                          ny=self.stamp_size[1],
                                          astrometric_error=self.astrometric_error,
                                          galaxies_distr_path=self.galaxies_distr_path,
                                          bands=self.bands,
                                          sky_clipping=self.sky_clipping,
                                          ccd_parameters=self.camera_params["CCD25"])

        self.load_obs_conditions()
        self.load_lightcurves()
        self.mag_to_counts()
        self.make_save_images()

    def load_obs_conditions(self):
        # This part is horrible, I'm sorry
        fields = list(self.obs_cond.keys())
        self.sorted_obs_cond = {}
        epoch_keys = list(self.obs_cond[fields[0]][0].keys())
        for field in fields:
            self.sorted_obs_cond[field] = {}
            epoch_list = self.obs_cond[field]
            for key in epoch_keys:
                self.sorted_obs_cond[field][key] = {}
                for band in ["g", "r", "i"]:
                    self.sorted_obs_cond[field][key][band] = []

            aux_epochs = self.obs_cond[field]
            for epoch in epoch_list:
                for key in epoch_keys:
                    self.sorted_obs_cond[field][key][epoch["filter"]].append(epoch[key])
            for band in self.bands:
                sorted_index = np.argsort(self.sorted_obs_cond[field]["obs_day"][band])
                for key in epoch_keys:
                    self.sorted_obs_cond[field][key][band] = np.array(self.sorted_obs_cond[field][key][band])[sorted_index]
            #print(list(self.sorted_obs_cond.keys()))
            #print(list(self.sorted_obs_cond["Field01"].keys()))
            #print(list(self.sorted_obs_cond["Field01"]["sky_brightness"].keys()))
            #for key in epoch_keys:
            #    print("-----"+key+"-----")
            #    for band in ["g", "r", "i"]:
            #        print("band: "+band)
            #        print(self.sorted_obs_cond["Field01"][key][band])



    def load_lightcurves(self):
        self.data = h5py.File(self.lc_path+".hdf5", "r")
        self.lc_params = np.load(self.lc_path+".pkl")
        self.fields = list(self.data.keys())
        print("light curve classes found")
        data_classes = list(self.data[list(self.data.keys())[0]]["n_lc_per_type"].keys())
        self.n_per_type = self.data[list(self.data.keys())[0]]["n_lc_per_type"]
        print(data_classes)
        print("n light curves found")
        self.n_lightcurves = len(list(self.data.keys()))*self.data[list(self.data.keys())[0]]["lc_id"].shape[0]
        print(self.n_lightcurves)
        self.lc_with_galaxies = {}
        self.lc_with_galaxies_count = {}
        self.type_count = {}
        for lc_type in data_classes:
            aux_index = self.requested_lightcurves.index(lc_type)
            n = self.n_per_type[lc_type][()]*self.prop_lightcurves_with_galaxies[aux_index]
            n = int(np.floor(n))
            aux = np.concatenate([np.ones(shape=(n,)),
                                  np.zeros(shape=(self.n_per_type[lc_type][()]-n,))],
                                 axis=0)
            np.random.shuffle(aux)
            self.lc_with_galaxies[lc_type] = aux
            self.type_count[lc_type] = 0

    def mag_to_counts(self, sky_from_SNLS = True, save_lc=False):
        print("From magnitudes to counts")
        self.c_lightcurves = {}
        for field in self.fields:
            self.c_lightcurves[field] = {}
            for band in self.bands:
                m_lightcurves = self.data[field]["lightcurves"][band][:]
                self.c_lightcurves[field][band] = Mag2Counts(lightcurves=m_lightcurves,
                                                             airmass_per_obs=None,
                                                             t_exp=self.sorted_obs_cond[field]["exp_time"][band],
                                                             zero_point=self.sorted_obs_cond[field]["zero_point"][band])
                print(self.c_lightcurves[field][band].shape)
                #plt.plot(self.sorted_obs_cond[field]["obs_day"]["g"], self.c_lightcurves[field][band][0,:])
                #plt.show()
        print(self.c_lightcurves["Field01"]["g"][:10, :10])

    # TODO: Only this part
    def make_save_images(self):

        # This version has no stacking option (for simplicity)
        print("producing and saving images")
        hdf5_file = h5py.File(self.save_path+self.output_filename+".hdf5", 'w')
        dt = h5py.special_dtype(vlen=str)
        field_group = {}
        for field in self.fields:
            field_group[field] = hdf5_file.create_group()
            hdf5_file.create_dataset("lc_type",
                                     shape=(0,),
                                     dtype=dt)
            hdf5_file.create_dataset("obs_cond",
                                     )
            hdf5_file.create_dataset("labels",
                                     shape=(0,))
            hdf5_file.create_dataset("ids",
                                     shape=(0,))



if __name__ == "__main__":

    save_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/"
    output_filename = "wena"
    bands = ["g", ]
    galaxy_path = "/home/rodrigo/supernovae_detection/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
    lightcurves_path = "/home/rodrigo/supernovae_detection/simulated_data/lightcurves/hits_100"
    camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
    requested_lightcurve = ["Supernovae", "RRLyrae", "M33Cepheids", "NonVariable", "EmptyLightCurve", "Asteroids"]
    requested_lightcurve_labels = [0, 1, 2, 3, 4, 5]  # multiclass
    stamp_size = (21, 21)
    proportion_with_galaxy = [0.5, 0.05, 0.05, 0.05, 0.5, 0]
    lc_per_chunk = 100
    sky_clipping = 2000
    astrometric_error = 0.3
    image_stacking_time = 0

    start = time.time()
    database = ImageDatabase(save_path=save_path,
                             output_filename=output_filename,
                             bands=bands,
                             galaxies_distr_path=galaxy_path,
                             lc_path=lightcurves_path,
                             cam_and_obs_cond_path=camera_and_obs_cond_path,
                             requested_lightcurves=requested_lightcurve,
                             requested_lightcurves_labels=requested_lightcurve_labels,
                             stamp_size=stamp_size,
                             prop_lightcurves_with_galaxies=proportion_with_galaxy,
                             lc_per_chunk=lc_per_chunk,
                             estimated_sky_clipping=sky_clipping,
                             astrometric_error=astrometric_error,
                             image_stacking_time=image_stacking_time)
    end = time.time()
    print("elapsed time: " + str(end - start))
    print("wena")
