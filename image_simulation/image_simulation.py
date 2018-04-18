import numpy as np
from read_SNLS import SNLSDatabase
from ImageFactory import ImageFactory
import h5py
from mag_to_counts import Mag2Counts
import time
import sys
import pickle


class SupernovaDatabase(object):

    def __init__(self, **kwargs):
        # Path
        self.save_path = kwargs["save_path"]
        self.output_filename = kwargs["output_filename"]
        self.bands = kwargs["bands"]
        self.SNLS_path = kwargs["SNLS_path"]
        self.limmag = kwargs["limmag"]
        self.days = kwargs["days"]
        self.galaxies_distr_path = kwargs["galaxies_distr_path"]
        self.lc_path = kwargs["lc_path"]
        # Light curves to simulate
        self.requested_lightcurves = kwargs["requested_lightcurves"]
        self.requested_lightcurves_labels = kwargs["requested_lightcurves_labels"]
        self.stamp_size = kwargs["stamp_size"]
        self.prop_lightcurves_with_galaxies = kwargs["prop_lightcurves_with_galaxies"]
        # Instrument and observation conditions
        self.zero_points = kwargs["zero_point"]
        self.exp_times = kwargs["exp_times"]
        self.lc_per_chunk = kwargs["lc_per_chunk"]
        self.airmass_term = kwargs["airmass_term"]
        self.readout_noise = kwargs["readout_noise"]
        self.gain = kwargs["gain"]
        self.pixel_saturation = kwargs["pixel_saturation"]
        self.pixel_scale = kwargs["pixel_scale"]
        self.sky_clipping = kwargs["estimated_sky_clipping"]
        # Light curves parameters
        self.astrometric_error = kwargs["astrometric_error"]
        self.image_stacking_time = kwargs["image_stacking_time"]

        self.SNLS_database = SNLSDatabase(lightcurve_directory=self.SNLS_path, load_pickle=True)
        self.image_factory = ImageFactory(nx=self.stamp_size[0],
                                          ny=self.stamp_size[1],
                                          readout_noise=self.readout_noise,
                                          gain=self.gain,
                                          pixel_saturation=self.pixel_saturation,
                                          astrometric_error=self.astrometric_error,
                                          pixel_size=self.pixel_scale,
                                          galaxies_distr_path=self.galaxies_distr_path,
                                          zero_point=self.zero_points,
                                          exp_times=self.exp_times,
                                          bands=self.bands,
                                          sky_clipping=self.sky_clipping,
                                          airmass_terms=self.airmass_term)

        self.observation_conditions()
        self.load_lightcurves()
        self.mag_to_counts()
        self.make_save_images()

    def observation_conditions(self):
        # it produces (self.days_SNLS, self.sky_SNLS)
        #print(self.lightcurves.shape[1])
        print("reading observation conditions from SNLS")
        self.observation_data = self.SNLS_database.filter_data
        self.seeing = {}
        self.sky_estimation = {}
        self.airmass = {}
        for band in self.bands:
            self.seeing[band] = self.observation_data[band]["seeing"][:len(self.days[band])]
            self.sky_estimation[band] = self.observation_data[band]["estimated_sky"][:len(self.days[band])]
            self.airmass[band] = self.observation_data[band]["airmass"][:len(self.days[band])]
        print(str(self.observation_data.keys()))
        print(str(self.observation_data[self.observation_data.keys()[0]].keys()))
        for i in self.sky_estimation.keys():
            print(self.sky_estimation[i][:5])

    def load_lightcurves(self):
        data = h5py.File(self.lc_path+".hdf5", "r")
        print(data.keys())
        m_lightcurves = data["lightcurves"]  # dictionary
        self.m_lightcurves = {}
        for band in self.bands:
            self.m_lightcurves[band] = m_lightcurves[band][:]
        n_per_type = data["n_lc_per_type"]
        self.n_per_type = {}
        for object_name in self.requested_lightcurves:
            self.n_per_type[object_name] = n_per_type[object_name][()]
        self.labels = data["labels"][:]
        self.lc_type = data["lc_type"][:]
        #self.params = data["parameters"]  # dictionary
        self.params = np.load(self.lc_path+".pkl")["parameters"]
        self.lc_id = data["lc_id"][:]
        self.n_lightcurves = len(self.labels)
        self.lc_with_galaxies = {}
        self.lc_with_galaxies_count = {}
        self.type_count = {}
        for lc_type in np.unique(self.lc_type):
            aux_index = self.requested_lightcurves.index(lc_type)
            n = self.n_per_type[lc_type]*self.prop_lightcurves_with_galaxies[aux_index]
            n = int(np.floor(n))
            aux = np.concatenate([np.ones(shape=(n,)),
                                  np.zeros(shape=(self.n_per_type[lc_type]-n,))],
                                 axis=0)
            np.random.shuffle(aux)
            self.lc_with_galaxies[lc_type] = aux
            self.type_count[lc_type] = 0

    def mag_to_counts(self, sky_from_SNLS = True, save_lc=False):
        print("From magnitudes to counts")
        self.c_lightcurves = {}
        for band in self.bands:
            self.c_lightcurves[band] = Mag2Counts(self.m_lightcurves[band],
                                                  self.airmass[band],
                                                  self.exp_times[band],
                                                  self.zero_points[band],
                                                  self.airmass_term[band])

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
    if sys.argv[1] == "SNLS":
        from config_images import *
    elif sys.argv[1] == "KMTNet":
        from config_images_KMTNet import *
    start = time.time()
    database = SupernovaDatabase(save_path=save_path,
                                 output_filename=output_filename,
                                 lc_path=lc_path,
                                 SNLS_path=SNLS_path,
                                 days=observation_days,
                                 limmag=limmag,
                                 galaxies_distr_path=galaxies_distr_path,
                                 requested_lightcurves=requested_lightcurve,
                                 requested_lightcurves_labels=requested_lightcurve_labels,
                                 prop_lightcurves_with_galaxies=prop_lightcurves_with_galaxies,
                                 pixel_scale=pixel_scale,
                                 zero_point=zero_points,
                                 exp_times=exp_times,
                                 airmass_term=airmass_term,
                                 stamp_size=stamp_size,
                                 readout_noise=readout_noise,
                                 gain=gain,
                                 pixel_saturation=pixel_saturation,
                                 lc_per_chunk=lc_per_chunk,
                                 estimated_sky_clipping=estimated_sky_clipping,
                                 astrometric_error=astrometric_error,
                                 bands=bands,
                                 image_stacking_time=image_stacking_time)
    end = time.time()
    print("elapsed time: " + str(end - start))
    print("wena")
