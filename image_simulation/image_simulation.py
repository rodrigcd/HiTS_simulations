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
        # self.make_save_images()

    # TODO: STEP BY STEP UNTIL EVERYTHING WORKS
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
                plt.plot(self.sorted_obs_cond[field]["obs_day"]["g"], self.c_lightcurves[field][band][0,:])
                plt.show()
        print(self.c_lightcurves["Field01"]["g"][:10, :10])

    # TODO: Only this part
    def make_save_images(self):

        def stack_images(images, gal_images, lighcurves, days, stacking_time):
            # Extracting consecutive images
            day_diff = np.diff(days)
            seconds_diff = day_diff * 24 * 60 * 60.0
            consec_index = np.where(seconds_diff < stacking_time)[0]
            index = np.split(consec_index, np.where(np.diff(consec_index) != 1)[0] + 1)
            #print(len(index))
            current_index = 0
            group_index = 0
            stacked_images = []
            stacked_gal = []
            stacked_days = []
            stacked_index = []
            while current_index < len(days) and group_index < len(index):
                if current_index not in index[group_index]:
                    stacked_images.append(images[:, :, :, current_index][..., np.newaxis])
                    stacked_gal.append(gal_images[:, :, :, current_index][..., np.newaxis])
                    stacked_days.append(days[current_index])
                    stacked_index.append(current_index)
                    current_index += 1
                    continue
                indexes = np.append(index[group_index], np.amax(index[group_index]) + 1)
                group_images = images[:, :, :, indexes]
                group_gal = gal_images[:, :, :, indexes]
                stacked_images.append(np.mean(group_images, axis=3)[..., np.newaxis])
                stacked_gal.append(np.mean(group_gal, axis=3)[..., np.newaxis])
                stacked_days.append(days[index[group_index][0]])
                stacked_index.append(current_index)
                current_index = np.amax(index[group_index]) + 2
                group_index += 1
            while current_index < len(days):
                stacked_images.append(images[:, :, :, current_index][..., np.newaxis])
                stacked_gal.append(gal_images[:, :, :, current_index][..., np.newaxis])
                stacked_days.append(days[current_index])
                stacked_index.append(current_index)
            stacked_images = np.concatenate(stacked_images, axis=3)
            stacked_gal = np.concatenate(stacked_gal, axis=3)
            stacked_lc = lighcurves[..., np.array(stacked_index)]
            stacked_days = np.array(stacked_days)
            #print(len(stacked_days))
            return stacked_images, stacked_gal, stacked_lc, stacked_days, np.array(stacked_index)

        print("producing and saving images")
        hdf5_file = h5py.File(self.save_path+self.output_filename+".hdf5", 'w')
        dt = h5py.special_dtype(vlen=unicode)
        hdf5_file.create_dataset("lc_type",
                                 data=self.lc_type,
                                 dtype=dt)
        hdf5_file.create_dataset("labels",
                                 data=self.labels)
        hdf5_file.create_dataset("ids",
                                 data=self.lc_id)

        band_groups = {}
        chunk_c_lc = {}
        chunk_m_lc = {}
        chunk_id = {}
        chunk_gal_flag = {}
        images = {}
        image_dset = {}
        gal_images = {}
        gal_images_dset = {}
        lc_dset = {}
        lc_counts_dset = {}
        ids_dset = {}
        galaxy_flag_dset = {}
        stacked_index_state = {}
        galaxy_parameters = {}
        galaxy_parameters["headers"] = self.image_factory.galaxies_gen.return_headers()
        galaxy_parameters["data"] = []

        for band in self.bands:
            band_groups[band] = hdf5_file.create_group(band)

            #if self.image_stacking_time == 0:
            band_groups[band].create_dataset("sky_estimation",
                                             data=self.sky_estimation[band])
            band_groups[band].create_dataset("observation_days",
                                             data=self.days[band])
            band_groups[band].create_dataset("limmag",
                                             data=self.limmag[band])
            band_groups[band].create_dataset("seeing",
                                             data=self.seeing[band])
            band_groups[band].create_dataset("airmass",
                                             data=self.airmass[band])
            band_groups[band].create_dataset("complete_counts_lc",
                                             data=self.c_lightcurves[band])
            image_dset[band] = band_groups[band].create_dataset("images",
                                                                shape=(0, self.stamp_size[0], self.stamp_size[1], 0),
                                                                maxshape=(None, self.stamp_size[0], self.stamp_size[1], None))
            gal_images_dset[band] = band_groups[band].create_dataset("gal_images",
                                                                     shape=(0, self.stamp_size[0], self.stamp_size[1], 0),
                                                                     maxshape=(None, self.stamp_size[0], self.stamp_size[1], None))
            lc_dset[band] = band_groups[band].create_dataset("lightcurves",
                                                             shape=(0, 0),
                                                             maxshape=(None, None))
            lc_counts_dset[band] = band_groups[band].create_dataset("counts_lightcurves",
                                                                    shape=(0, 0),
                                                                    maxshape=(None, None))
            ids_dset[band] = band_groups[band].create_dataset("ids",
                                                              shape=(0,),
                                                              maxshape=(None,))
            galaxy_flag_dset[band] = band_groups[band].create_dataset("galaxy_flag",
                                                                      shape=(0,),
                                                                      maxshape=(None,))
            stacked_index_state[band] = True
            chunk_m_lc[band] = []
            chunk_c_lc[band] = []
            chunk_id[band] = []
            chunk_gal_flag[band] = []
            images[band] = []
            gal_images[band] = []

        count = 0
        n_chunk = 0
        aux_too_many = False
        for i in range(self.labels.shape[0]):

            # Add or not add galaxy

            with_galaxy = self.lc_with_galaxies[self.lc_type[i]][self.type_count[self.lc_type[i]]] == 1
            self.type_count[self.lc_type[i]] += 1
            redshift = []
            if with_galaxy:
                if self.lc_type[i] == "Supernovae":
                    redshift = np.exp(self.params[band][i])  # log redshift

            aux_c_lc = {}
            for band in self.bands:
                aux_c_lc[band] = self.c_lightcurves[band][np.newaxis, i, :]
                chunk_c_lc[band].append(self.c_lightcurves[band][np.newaxis, i, :])
                chunk_m_lc[band].append(self.m_lightcurves[band][np.newaxis, i, :])
                chunk_id[band].append(self.lc_id[i])
                chunk_gal_flag[band].append(with_galaxy)

            image, gal_image = self.image_factory.createLightCurveImages(counts=aux_c_lc,
                                                                         seeing=self.seeing,
                                                                         airmass=self.airmass,
                                                                         sky_counts=self.sky_estimation,
                                                                         zero_point=self.zero_points,
                                                                         redshift=redshift,
                                                                         with_galaxy=with_galaxy)

            if with_galaxy:
                galaxy_parameters["data"].append(self.image_factory.galaxies_gen.return_params())
            else:
                galaxy_parameters["data"].append([])

            for band in self.bands:
                images[band].append(image[band][np.newaxis, ...])
                gal_images[band].append(gal_image[band][np.newaxis, ...])

            count += 1
            if count >= self.lc_per_chunk or (i == self.labels.shape[0] - 1):
                print("chunk" + str(n_chunk + 1) + " of " + str(self.n_lightcurves / (self.lc_per_chunk * 1.0)))
                for band in self.bands:
                    images[band] = np.concatenate(images[band], axis=0)
                    gal_images[band] = np.concatenate(gal_images[band], axis=0)
                    chunk_c_lc[band] = np.concatenate(chunk_c_lc[band], axis=0)
                    chunk_m_lc[band] = np.concatenate(chunk_m_lc[band], axis=0)
                    chunk_id[band] = np.array(chunk_id[band])
                    chunk_gal_flag[band] = np.array(chunk_gal_flag[band])
                    # Saving group
                    if self.image_stacking_time > 0:
                        images[band], gal_images[band], chunk_m_lc[band], days, index = stack_images(images[band],
                                                                                                     gal_images[band],
                                                                                                     chunk_m_lc[band],
                                                                                                     self.days[band],
                                                                                                     self.image_stacking_time)
                        chunk_c_lc[band] = chunk_c_lc[band][..., index]
                        if stacked_index_state[band]:
                            print("stacking images under " + str(self.image_stacking_time) + "s of cadence")
                            band_groups[band].create_dataset("stacked_sky_estimation", data=self.sky_estimation[band][index])
                            band_groups[band].create_dataset("stacked_observation_days", data=self.days[band][index])
                            band_groups[band].create_dataset("stacked_limmag", data=self.limmag[band][index])
                            band_groups[band].create_dataset("stacked_seeing", data=self.seeing[band][index])
                            band_groups[band].create_dataset("stacked_airmass", data=self.airmass[band][index])
                            band_groups[band].create_dataset("stacked_index", data=index)
                            stacked_index_state[band] = False
                    # images
                    prev_index = image_dset[band].shape[0]
                    image_dset[band].shape = (prev_index+images[band].shape[0],
                                              images[band].shape[1],
                                              images[band].shape[2],
                                              images[band].shape[3])
                    image_dset[band][prev_index:, :, :, :] = images[band]

                    # gal_images
                    prev_index = gal_images_dset[band].shape[0]
                    gal_images_dset[band].shape = (prev_index+gal_images[band].shape[0],
                                                   gal_images[band].shape[1],
                                                   gal_images[band].shape[2],
                                                   gal_images[band].shape[3])
                    gal_images_dset[band][prev_index:, :, :, :] = gal_images[band]

                    # lightcurves
                    lc_dset[band].shape = (prev_index+chunk_m_lc[band].shape[0],
                                           chunk_m_lc[band].shape[1])
                    lc_dset[band][prev_index:, :] = chunk_m_lc[band]
                    lc_counts_dset[band].shape = (prev_index+chunk_c_lc[band].shape[0],
                                                  chunk_c_lc[band].shape[1])
                    lc_counts_dset[band][prev_index:, :] = chunk_c_lc[band]

                    # class and type
                    ids_dset[band].shape = (prev_index+chunk_id[band].shape[0],)
                    ids_dset[band][prev_index:] = chunk_id[band]
                    galaxy_flag_dset[band].shape = (prev_index+chunk_gal_flag[band].shape[0],)
                    galaxy_flag_dset[band][prev_index:] = chunk_gal_flag[band]

                    images[band] = []
                    gal_images[band] = []
                    chunk_c_lc[band] = []
                    chunk_m_lc[band] = []
                    chunk_id[band] = []
                    chunk_gal_flag[band] = []

                n_chunk += 1
                count = 0

        pickle.dump(galaxy_parameters,
                    open(self.save_path+self.output_filename+"_galaxies_params.pkl", "wb"),
                    protocol=2)


if __name__ == "__main__":

    save_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/"
    output_filename = "wena"
    bands = ["g",]
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
