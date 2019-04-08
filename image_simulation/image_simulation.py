import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from ImageFactory import ImageFactory
import h5py
from MagCounts import Mag2Counts
import time
import random
import sys
import pickle
import matplotlib.pyplot as plt


class ImageDatabase(object):

    def __init__(self, **kwargs):
        # Path
        print("----- Running image simulation -----")
        self.save_path = kwargs["save_path"]
        print("Save path: "+self.save_path)
        self.output_filename = kwargs["output_filename"]
        self.bands = kwargs["bands"]
        self.galaxies_distr_path = kwargs["galaxies_distr_path"]
        self.lc_path = kwargs["lc_path"]
        print("Light curves path: "+self.lc_path)
        self.camera_and_obs_cond_path = kwargs["cam_and_obs_cond_path"]
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
        self.augmented_psfs = kwargs["augmented_psf"]
        # self.image_stacking_time = kwargs["image_stacking_time"]

        self.load_obs_conditions()
        self.load_image_factory()
        self.load_lightcurves()
        self.reset_galaxy_counts()
        self.mag_to_counts()
        self.make_save_images()

    def load_image_factory(self):
        print("- Image Factory")
        self.image_factory = ImageFactory(nx=self.stamp_size[0],
                                          ny=self.stamp_size[1],
                                          astrometric_error=self.astrometric_error,
                                          galaxies_distr_path=self.galaxies_distr_path,
                                          bands=self.bands,
                                          sky_clipping=self.sky_clipping,
                                          ccd_parameters=self.camera_params[list(self.camera_params.keys())[0]],
                                          real_psfs=True,
                                          obs_cond_path=self.camera_and_obs_cond_path,
                                          augmented_psfs=self.augmented_psfs)

    def load_obs_conditions(self):
        # This part is horrible, I'm sorry
        fields = list(self.obs_cond.keys())
        sorted_obs_cond = {}
        epoch_keys = list(self.obs_cond[fields[0]][0].keys())
        for field in fields:
            sorted_obs_cond[field] = {}
            epoch_list = self.obs_cond[field]
            for key in epoch_keys:
                sorted_obs_cond[field][key] = {}
                for band in ["g", "r", "i"]:
                    sorted_obs_cond[field][key][band] = []

            for epoch in epoch_list:
                for key in epoch_keys:
                    #if key == "seeing":
                        # print(epoch["seeing"])
                    sorted_obs_cond[field][key][epoch["filter"]].append(epoch[key])
            for band in self.bands:
                sorted_index = np.argsort(sorted_obs_cond[field]["obs_day"][band])
                for key in epoch_keys:
                    sorted_obs_cond[field][key][band] = np.array(sorted_obs_cond[field][key][band])[sorted_index]
            #print(list(sorted_obs_cond.keys()))
            #print(list(sorted_obs_cond["Field01"].keys()))
            #print(list(sorted_obs_cond["Field01"]["sky_brightness"].keys()))
            #for key in epoch_keys:
            #    print("-----"+key+"-----")
            #    for band in ["g", "r", "i"]:
            #        print("band: "+band)
            #        print(sorted_obs_cond["Field01"][key][band])
        self.obs_cond = sorted_obs_cond
        # print(type(self.obs_cond))
        # print(self.obs_cond.keys())
        # print(self.obs_cond["Field01"].keys())
        # for key in self.obs_cond["Field01"].keys():
        #     print(key)
        #     print(self.obs_cond["Field01"][key].keys())
        # print(self.obs_cond["Field01"]["filter"]["g"])



    def load_lightcurves(self):
        self.data = h5py.File(self.lc_path+".hdf5", "r")
        self.lc_params = np.load(self.lc_path+".pkl")
        self.fields = list(self.data.keys())
        print("- Light curve classes found")
        self.data_classes = list(self.data[list(self.data.keys())[0]]["n_lc_per_type"].keys())
        self.n_per_type = self.data[list(self.data.keys())[0]]["n_lc_per_type"]
        print(self.data_classes)
        #print("n light curves found")
        self.n_lightcurves = len(list(self.data.keys()))*self.data[list(self.data.keys())[0]]["lc_id"].shape[0]
        print("- Total number of lightcurves"+str(self.n_lightcurves))

    def reset_galaxy_counts(self):

        self.lc_with_galaxies = {}
        self.lc_with_galaxies_count = {}
        self.type_count = {}

        for lc_type in self.data_classes:
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
        print("- From magnitudes to counts")
        self.c_lightcurves = {}
        for field in self.fields:
            self.c_lightcurves[field] = {}
            for band in self.bands:
                m_lightcurves = self.data[field]["lightcurves"][band][:]
                self.c_lightcurves[field][band] = Mag2Counts(lightcurves=m_lightcurves,
                                                             airmass_per_obs=None,
                                                             t_exp=self.obs_cond[field]["exp_time"][band],
                                                             zero_point=self.obs_cond[field]["zero_point"][band])
                #print(self.c_lightcurves[field][band].shape)
                #plt.plot(self.sorted_obs_cond[field]["obs_day"]["g"], self.c_lightcurves[field][band][0,:])
                #plt.show()
        #print(self.c_lightcurves["Field01"]["g"][:10, :10])

    # TODO: Only this part
    def make_save_images(self):

        # This version has no stacking option (for simplicity)
        print("- Producing and Saving images")
        hdf5_file = h5py.File(self.save_path+self.output_filename+".hdf5", 'w')
        dt = h5py.special_dtype(vlen=str)

        field_group = {}
        lc_type_dataset = {}
        labels_dataset = {}
        ids_dataset = {}
        galaxy_flag_dataset = {}
        image_dset = {}
        lc_dset = {}
        lc_count_dset = {}
        galaxy_dset = {}
        psf_dset = {}

        for i_field, field in enumerate(self.fields):
            start_time = time.time()
            print("Simulating field "+field)
            self.reset_galaxy_counts()

            for lc_type in self.data_classes:
                self.type_count[lc_type] = 0

            n_field_lightcurves = len(self.data[field]["labels"])

            field_group[field] = hdf5_file.create_group(name=field)
            lc_type_dataset[field] = field_group[field].create_dataset("lc_type",
                                                                       data=self.data[field]["lc_type"][:],
                                                                       dtype=dt)
            labels_dataset[field] = field_group[field].create_dataset("labels",
                                                                      data=self.data[field]["labels"][:])
            ids_dataset[field] = field_group[field].create_dataset("ids",
                                                                   data=self.data[field]["lc_id"])
            galaxy_flag_dataset[field] = field_group[field].create_dataset("galaxy_flag",
                                                                           shape=(n_field_lightcurves,))
            # Saving obs cond
            obs_group = field_group[field].create_group(name="obs_cond")
            #for key1 in self.obs_cond.keys():
            #    obs_feature_group = obs_group.create_group(name=key1)
            for key2 in self.obs_cond[field].keys():
                band_group = obs_group.create_group(name=key2)
                if key2 == "filter":
                    continue
                if key2 == "limmag3":
                    continue
                for band in self.bands:
                    #print(field, key2, band)
                    #print(self.obs_cond[field][key2][band])
                    band_group.create_dataset(name=band,
                                              data=self.obs_cond[field][key2][band])
                    if key2 == "seeing":
                        print(self.obs_cond[field][key2][band])

            image_dset[field] = {}
            galaxy_dset[field] = {}
            psf_dset[field] = {}
            lc_dset[field] = {}
            lc_count_dset[field] = {}

            image_group = field_group[field].create_group(name="images")
            gal_image_group = field_group[field].create_group(name="galaxy_image")
            psf_group = field_group[field].create_group(name="psf_image")
            lc_group = field_group[field].create_group(name="lightcurves")
            lc_count_group = field_group[field].create_group(name="count_lightcurves")

            for band in self.bands:

                n_obs = len(self.data[field]["obs_days"][band])
                image_dset[band] = image_group.create_dataset(name=band, shape=(n_field_lightcurves,
                                                                                self.stamp_size[0],
                                                                                self.stamp_size[1],
                                                                                n_obs))
                galaxy_dset[band] = gal_image_group.create_dataset(name=band, shape=(n_field_lightcurves,
                                                                                     self.stamp_size[0],
                                                                                     self.stamp_size[1],
                                                                                     n_obs))
                psf_dset[band] = psf_group.create_dataset(name=band, shape=(n_field_lightcurves,
                                                                            self.stamp_size[0],
                                                                            self.stamp_size[1],
                                                                            n_obs))
                lc_dset[band] = lc_group.create_dataset(name=band, shape=(n_field_lightcurves, n_obs))
                lc_count_dset[band] = lc_count_group.create_dataset(name=band, shape=(n_field_lightcurves, n_obs))

            n_saved_lightcurves = 0

            while n_saved_lightcurves < n_field_lightcurves:

                image_chunk = {}
                psf_chunk = {}
                galaxy_image_chunk = {}
                for band in self.bands:
                    image_chunk[band] = []
                    psf_chunk[band] = []
                    galaxy_image_chunk[band] = []

                upper_index = n_saved_lightcurves + self.lc_per_chunk

                if upper_index > n_field_lightcurves:
                    upper_index = n_field_lightcurves

                current_types = self.data[field]["lc_type"][n_saved_lightcurves:upper_index]
                current_params = self.lc_params[field][band][n_saved_lightcurves:upper_index]

                #print(type(self.camera_params))

                seeing = self.obs_cond[field]["seeing"]
                zero_point = self.obs_cond[field]["zero_point"]
                sky_brightness = self.obs_cond[field]["sky_brightness"]
                exp_time = self.obs_cond[field]["exp_time"]

                galaxy_flag_array = []

                for lc_i in range(upper_index-n_saved_lightcurves):

                    random_camera = self.camera_params[random.choice(list(self.camera_params.keys()))]
                    self.image_factory.set_ccd_params(random_camera)

                    with_galaxy = self.lc_with_galaxies[current_types[lc_i]][self.type_count[current_types[lc_i]]] == 1
                    galaxy_flag_array.append(with_galaxy)

                    self.type_count[current_types[lc_i]] += 1
                    redshift = []
                    if with_galaxy:
                        if current_types[lc_i] == "Supernovae":
                            redshift = np.exp(current_params[lc_i][0])  # log redshift
                            #print(redshift)

                    aux_c_lc = {}
                    for band in self.bands:
                        aux_c_lc[band] = self.c_lightcurves[field][band][n_saved_lightcurves:upper_index, :]
                        aux_c_lc[band] = aux_c_lc[band][np.newaxis, lc_i, :]

                    image, gal_image, psf = self.image_factory.createLightCurveImages(counts=aux_c_lc,
                                                                                      seeing=seeing,
                                                                                      airmass=None,
                                                                                      sky_counts=sky_brightness,
                                                                                      exp_time=exp_time,
                                                                                      zero_point=zero_point,
                                                                                      redshift=redshift,
                                                                                      with_galaxy=with_galaxy)

                    for band in self.bands:
                        # print(image[band])
                        image_chunk[band].append(image[band])
                        galaxy_image_chunk[band].append(gal_image[band])
                        psf_chunk[band].append(psf[band])

                for i_band, band in enumerate(self.bands):
                    current_mag_lc = self.data[field]["lightcurves"][band][n_saved_lightcurves:upper_index, :]
                    current_lc = self.c_lightcurves[field][band][n_saved_lightcurves:upper_index, :]
                    image_chunk[band] = np.stack(image_chunk[band])
                    print("image sequence shape for band "+band+": "+str(image_chunk[band].shape))
                    galaxy_image_chunk[band] = np.stack(galaxy_image_chunk[band])
                    psf_chunk[band] = np.stack(psf_chunk[band])
                    image_dset[band][n_saved_lightcurves:upper_index, :, :, :] = image_chunk[band]
                    galaxy_dset[band][n_saved_lightcurves:upper_index, :, :, :] = galaxy_image_chunk[band]
                    psf_dset[band][n_saved_lightcurves:upper_index, :, :, :] = psf_chunk[band]
                    lc_dset[band][n_saved_lightcurves:upper_index, :] = current_mag_lc
                    lc_count_dset[band][n_saved_lightcurves:upper_index, :] = current_lc
                galaxy_flag_dataset[field][n_saved_lightcurves:upper_index] = np.array(galaxy_flag_array)

                n_saved_lightcurves = upper_index

            field_end_time = time.time()
            print("Elapsed time per field: "+str("%.2f" % (field_end_time-start_time))+"s")

    def filter_by_conditions(self, condition_limits):
        hdf5_file = h5py.File(self.save_path + self.output_filename + ".hdf5", 'r+')
        fields = list(hdf5_file.keys())

        for field in fields:
            obs_cond_group = hdf5_file[field]["obs_cond"]
            if "good_quality_points" in list(obs_cond_group.keys()):
                del obs_cond_group["good_quality_points"]
            point_quality_gruop = obs_cond_group.create_group(name="good_quality_points")
            for band in self.bands:
                good_quality_points = np.ones(shape=obs_cond_group["obs_day"][band][:].shape)
                print("Filtering " + field + " band " +band +" with "+str(len(good_quality_points))+" points")
                for key, value in condition_limits.items():
                    cond = obs_cond_group[key][band][:]
                    cond_quality_points = np.logical_and(cond > value[band][0], cond < value[band][1])
                    good_quality_points = np.logical_and(good_quality_points, cond_quality_points)
                print(str(np.sum(good_quality_points)) + " points after filtering")
                point_quality_gruop.create_dataset(name=band, data=good_quality_points)

        hdf5_file.close()

    def generate_flux_conversion(self):
        hdf5_file = h5py.File(self.save_path + self.output_filename + ".hdf5", 'r+')
        fields = list(hdf5_file.keys())

        for field in fields:
            obs_cond_group = hdf5_file[field]["obs_cond"]
            if "flux_conversion" in list(obs_cond_group.keys()):
                del obs_cond_group["flux_conversion"]
            conversion_group = obs_cond_group.create_group(name="flux_conversion")
            for band in self.bands:
                band_zp = obs_cond_group["zero_point"][band][:]
                reference_zp = band_zp[0]
                flux_conversion = np.power(10, (reference_zp - band_zp)/2.5)
                print(flux_conversion)
                conversion_group.create_dataset(name=band, data=flux_conversion)

        hdf5_file.close()

if __name__ == "__main__":

    from config_images import *

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
                             augmented_psf=augmented_psfs)

    database.filter_by_conditions(filter_by_conditions)
    database.generate_flux_conversion()
    end = time.time()
    print("Total elapsed time: " + str(end - start))
    print("wena")
