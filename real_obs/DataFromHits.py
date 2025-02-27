import numpy as np
from astropy.io import fits as pf
import pickle
from numpy.linalg import norm
from os import listdir, walk
import os
import glob
from os.path import isfile, join, isdir
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.interpolation import rotate
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
import time

class HiTSData(object):

    CCDs_directory = '/home/toshiba/rodrigo/R2015CCDs/'
    target_image = 'science'
    #CCDs_list = [f for f in listdir(CCDs_directory) if isdir(join(CCDs_directory, f))]
    CCDs_list = [x for x in next(walk(CCDs_directory))[1]]
    CCDs_fits_list = list()
    CCDs_base_list = list()
    CCDs_diff_list = list()
    psf_list = list()
    for CCD in CCDs_list:
        fits_directory = CCDs_directory + CCD + '/' + target_image + '/'
        diff_directory = CCDs_directory + CCD + '/diff/'
        base_directory = CCDs_directory + CCD + '/base/'
        psf_directory = CCDs_directory + CCD + "psf"
        #ccds list of fits files
        #print(fits_directory+"*.fits")
        CCDs_fits_list.append(sorted(glob.glob(fits_directory+"*.fits"), key=str.lower))
        CCDs_diff_list.append(sorted(glob.glob(diff_directory+"*.fits"), key=str.lower))
        CCDs_diff_list.append(sorted(glob.glob(diff_directory + "*.fits"), key=str.lower))
        #CCDs_fits_list.append(sorted([fits_directory + f for f in listdir(fits_directory) if isfile(join(fits_directory, f))], key = str.lower))
        #base image of each ccd
        #print(glob.glob(base_directory+"*.fits"))
        CCDs_base_list.append(glob.glob(base_directory+"*.fits"))
        #asdasd
        #CCDs_base_list.append([base_directory + f for f in listdir(base_directory) if isfile(join(base_directory, f))])

    def __init__(self, extraction_method = 'sextractor'):
        #extraction_method = 'stride' or 'noise_level'
        self.extraction_method = extraction_method
        self.current_ccd = 0
        self.stamp_size = 21
        self.image_size = np.array([4096, 2048])
        self.n_epochs = 24
        self.get_sn_sequence() #it produces supernovae stamps and save it in pkl file
        #self.supernovae_augmentation()

        #self.get_catalog()
        #self.generate_images()

    def get_catalog(self):
        #generate sample list of numpy array with shape (xsize, ysize, epoch)
        self.epoch_objects = []
        self.epoch_images = []
        self.catalogs = Catalog(self.image_size, self.extraction_method)
        #if self.current_ccd == len(self.CCDs_list):
        #    self.current_ccd = 0
        #self.CCD = self.CCDs_list[self.current_ccd]
        #for i, fits in enumerate(self.CCDs_fits_list[self.current_ccd]):
        #    epoch_id = fits[:-5]
        #    hdulist = pf.open(fits)
        #    fits_file = hdulist[0]
            



    def generate_images(self, catalog):
        self.sequence_database = []
        half_image = int(np.floor(self.image_size/2))
        print(self.epoch_images.shape)
        for cat_object in catalog:
            x_limits = [cat_object[0] - half_image, cat_object[0] + half_image + 1]
            y_limits = [cat_object[1] - half_image, cat_object[1] + half_image + 1]
            stamp = self.epoch_images[x_limits[0]:x_limits[1], y_limits[0]:y_limits[1], :]
            stamp = np.reshape(stamp, (1, stamp.shape[0], stamp.shape[1], self.epoch_images.shape[2]))
            self.sequence_database.append(stamp)
        self.sequence_database = np.concatenate(self.sequence_database, axis = 0)
        self.current_ccd += 1




    # Get supernovae sequences to stamps, then augmentation
    def get_sn_sequence(self):
        #produce original stamps
        def load_sn_coordinates():
            self.sn_info = []
            exceptions = ["HiTS42SN", "HiTS66SN", "HiTS67SN"]
            with open("./hits_tables/ResultsTable2015.csv", 'r') as f_aux:
                for i in range(90):
                    sn_dict = {}
                    line = f_aux.readline().split(",")
                    sn_dict["date"] = line[1]
                    sn_dict["id"] = line[2]
                    sn_dict["pos"] = np.array([int(line[5]), int(line[6])])
                    if sn_dict["id"] in exceptions:
                        continue
                    self.sn_info.append(sn_dict)

        def add_objects(sn_dict, fits_list, diff_list, psf_list):
            stamps = []
            mjd_obs = []
            sky_brigtness = []
            sky_sigma = []
            airmass = []
            CCD_num = []
            exp_time = []
            gain = []
            seeing = []
            pixel_scale = []
            read_noise = []
            saturation_value = []
            headers_dict = {}
            diff_stamps = []
            psfs = []

            print(sn_dict)
            #x_limits = [sn_dict["pos"][0] - int(half_image[1]), sn_dict["pos"][0] + int(half_image[1]) + 1]
            x_limits = [sn_dict["pos"][0] - int(self.stamp_size/2), sn_dict["pos"][0] + int(self.stamp_size/2) + 1]
            #y_limits = [sn_dict["pos"][1] - int(half_image[0]), sn_dict["pos"][1] + int(half_image[0]) + 1]
            y_limits = [sn_dict["pos"][1] - int(self.stamp_size / 2), sn_dict["pos"][1] + int(self.stamp_size / 2) + 1]
            #print(x_limits)
            #print(sn_dict["pos"][0])
            for i, fits in enumerate(fits_list):
                #search objects on each fits file, then match objects
                hdulist = pf.open(fits, memmap=False)
                fits_file = hdulist[0]

                mjd_obs.append(fits_file.header["MJD-OBS"])
                #sky_brigtness.append(fits_file.header["SKYBRITE"])
                sky_sigma.append(fits_file.header["SKYSIGMA"])
                airmass.append(fits_file.header["AIRMASS"])
                CCD_num.append(fits_file.header["CCDNUM"])
                exp_time.append(fits_file.header["EXPTIME"])
                gain.append(fits_file.header["GAINA"])
                seeing.append(fits_file.header["FWHM"])
                pixel_scale.append(fits_file.header["PIXSCAL1"])
                read_noise.append(fits_file.header["RDNOISEA"])
                saturation_value.append(fits_file.header["SATURATA"])
                print(fits_file.header["FILTER"])

                epoch_image = fits_file.data

                aux_mean, aux_std = np.mean(epoch_image), np.std(epoch_image)
                n_iter = 1
                valid_range = epoch_image
                for j in range(n_iter):
                    valid_range = valid_range[valid_range > (aux_mean-3*aux_std)]
                    valid_range = valid_range[valid_range < (aux_mean+3*aux_std)]

                sky_brigtness.append(np.mean(valid_range))


                #print(sky_brigtness[-1])

                #os.system("ds9 "+fits)
                #asdasd

                stamp = epoch_image[x_limits[0]:x_limits[1], y_limits[0]:y_limits[1]]
                stamps.append(stamp)
                hdulist.close()
                del hdulist[0].data
                del fits_file
                del epoch_image

            stamps = np.stack(stamps, axis=2)
            print(stamps.shape)
            #asdasd
            #print(mjd_obs)
            #self.sn_sequences.append(stamp)
            #del epoch_images

            headers_dict["obs_days"] = np.sort(np.array(mjd_obs).astype(np.float))
            headers_dict["sky_brightness"] = np.array(sky_brigtness)
            headers_dict["sky_sigma"] = np.array(sky_sigma)
            headers_dict["airmass"] = np.array(airmass).astype(np.float)
            headers_dict["ccd_num"] = np.array(CCD_num)
            headers_dict["exp_time"] = np.array(exp_time).astype(np.float)
            headers_dict["gain"] = np.array(gain)
            headers_dict["seeing"] = np.array(seeing)
            headers_dict["pixel_scale"] = np.array(pixel_scale)
            headers_dict["read_noise"] = np.array(read_noise)
            headers_dict["saturation"] = np.array(saturation_value)

            for i, fits in enumerate(diff_list):
                hdu_diff = pf.open(diff_list[i], memmap=False)
                diff_data = hdu_diff[0].data
                diff_stamp = diff_data[x_limits[0]:x_limits[1], y_limits[0]:y_limits[1]]
                diff_stamps.append(diff_stamp)
                hdu_diff.close()
                del hdu_diff[0].data
                del hdu_diff
                del diff_data
            diff_stamps = np.stack(diff_stamps, axis=2)
            print(diff_stamps.shape)

            for i, psf_file in enumerate(psf_list):
                psf = np.load(psf_file)
                psfs.append(psf)
            psfs = np.stack(psfs, axis=2)
            print(psfs.shape)

            return stamps, diff_stamps, headers_dict, psfs

        load_sn_coordinates()
        self.sn_data = {}
        not_found = []
        for sn_dict in self.sn_info:
            #try:
            fits_directory = self.CCDs_directory + sn_dict["id"] + '/' + self.target_image + '/'
            base_directory = self.CCDs_directory + sn_dict["id"] + '/base/'
            diff_directory = self.CCDs_directory + sn_dict["id"] + '/diff/'
            psf_directory = self.CCDs_directory + sn_dict["id"] + "/psf/"
            fits_list = []
            base_list = []
            print(sn_dict["id"])
            #try:

            fits_list = glob.glob(fits_directory+"*.fits")
            fits_list = sorted(fits_list, key=str.lower)
            base_list = glob.glob(base_directory + "*.fits")
            fits_list = base_list + fits_list
            diff_list = sorted(glob.glob(diff_directory+"*.fits"))
            psf_list = sorted(glob.glob(psf_directory+"*.npy"))
            print("n epochs: "+str(len(fits_list)))
            stamp, diff_stamp, headers, psfs = add_objects(sn_dict, fits_list, diff_list, psf_list)
            self.sn_data[sn_dict["id"]] = {"images": stamp, "diff": diff_stamp, "headers": headers, "psf": psfs}

        #with open(self.CCDs_directory + 'sn_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        with open('sn_data.pkl', 'wb') as f:
            pickle.dump(self.sn_data, f)

    def supernovae_augmentation(self):
        self.sn_info = np.load(self.CCDs_directory + 'sn_data.pkl')
        self.sn_sequences = self.sn_info["data"]
        self.sn_mjd_obs = self.sn_info["mjd-obs"]
        self.sn_id = self.sn_info["id"]
        print("shapes before augmentation")
        print("sequences shape = " + str(self.sn_sequences.shape))
        print("sn_mjd_obs shape = " + str(self.sn_mjd_obs.shape))
        print("len sn_id = " + str(len(self.sn_id)))

        def rotate_sequences():
            angles = [90, 180, 270]
            rotated_images = []
            rotated_ids = []
            for angle in angles:
                rotated_images.append(rotate(self.sn_sequences, angle, axes = (2,1)))
                for sn_id in self.sn_id:
                    rotated_ids.append(sn_id + "_rot_"+str(angle))

            self.sn_sequences = np.concatenate([self.sn_sequences, np.concatenate(rotated_images, axis = 0)], axis = 0)
            self.sn_id = self.sn_id + rotated_ids
            self.sn_mjd_obs = np.tile(self.sn_mjd_obs, (len(angles) + 1,1))
            print("shapes after rotation")
            print("sequences shape = " + str(self.sn_sequences.shape))
            print("sn_mjd_obs shape = " + str(self.sn_mjd_obs.shape))
            print("len sn_id = " + str(len(self.sn_id)))

        def flip_sequences():
            flipped_sequences = []
            flipped_ids = []
            axes = [1,2]
            for axis in axes:
                flipped_sequences.append(np.flip(self.sn_sequences, axis = axis))
                for sn_id in self.sn_id:
                    flipped_ids.append(sn_id + "_flp_" + str(axis))

            self.sn_sequences = np.concatenate([self.sn_sequences, np.concatenate(flipped_sequences, axis = 0)], axis = 0)
            self.sn_id = self.sn_id + flipped_ids
            self.sn_mjd_obs = np.tile(self.sn_mjd_obs, (len(axes) + 1,1))
            print("shapes after flip")
            print("sequences shape = " + str(self.sn_sequences.shape))
            print("sn_mjd_obs shape = " + str(self.sn_mjd_obs.shape))
            print("len sn_id = " + str(len(self.sn_id)))

        def blur_sequences():

            def blur_tensor(sigma):
                sequences = []
                exceptions = []
                for sample in range(self.sn_sequences.shape[0]):
                    epoch_images = []
                    next_sequence = 0
                    for epoch in range(self.n_epochs):
                        aux_image = self.sn_sequences[sample,:,:,epoch]
                        #min_image = np.amin(aux_image)
                        #aux_image = aux_image - min_image
                        max_image = np.amax(aux_image)
                        if max_image != 0:
                            aux_image = aux_image/max_image
                        else:
                            next_sequence = 1
                            exceptions.append(sample)
                            print("skiping sequence")
                            break
                        aux_image = rescale_intensity(gaussian(aux_image, sigma=sigma))
                        aux_image = aux_image*max_image# + min_image
                        epoch_images.append(aux_image[...,np.newaxis])
                    if next_sequence:
                        continue
                    epoch_images = np.concatenate(epoch_images, axis = 2)
                    sequences.append(epoch_images[np.newaxis,...])
                sequences = np.concatenate(sequences, axis = 0)
                return sequences, exceptions

            blurred_sequences = []
            blurred_ids = []
            blurred_mjd = []
            sigmas = [0.5, 1.0, 1.5]
            for sigma in sigmas:
                bl_seq, exceptions = blur_tensor(sigma)
                blurred_sequences.append(bl_seq)
                blurred_mjd.append(np.delete(self.sn_mjd_obs, exceptions, 0))
                for i, sn_id in enumerate(self.sn_id):
                    if i in exceptions:
                        continue
                    blurred_ids.append(sn_id + "_blr_" + str(sigma))
            print(np.amax(self.sn_sequences))
            print(np.amax(np.concatenate(blurred_sequences, axis = 0)))

            self.sn_sequences = np.concatenate([self.sn_sequences, np.concatenate(blurred_sequences, axis = 0)], axis = 0)
            self.sn_id = self.sn_id + blurred_ids
            self.sn_mjd_obs = np.concatenate([self.sn_mjd_obs, np.concatenate(blurred_mjd, axis = 0)], axis = 0)
            print("shapes after blurring")
            print("sequences shape = " + str(self.sn_sequences.shape))
            print("sn_mjd_obs shape = " + str(self.sn_mjd_obs.shape))
            print("len sn_id = " + str(len(self.sn_id)))

        def add_noise_to_sequences():

            def add_noise(sigma, mode):
                sequences = []
                exceptions = []
                for sample in range(self.sn_sequences.shape[0]):
                    epoch_images = []
                    next_sequence = 0
                    for epoch in range(self.n_epochs):
                        aux_image = self.sn_sequences[sample,:,:,epoch]
                        #min_image = np.amin(aux_image)
                        #aux_image = aux_image - min_image
                        max_image = np.amax(aux_image)
                        if max_image != 0:
                            aux_image = aux_image/max_image
                        else:
                            next_sequence = 1
                            exceptions.append(sample)
                            print("skiping sequence")
                            break
                        if mode == "poisson":
                            aux_image = random_noise(aux_image, mode = mode)
                        else:
                            aux_image = random_noise(aux_image, mode = mode, var=sigma)
                        aux_image = aux_image*max_image# + min_image
                        epoch_images.append(aux_image[...,np.newaxis])
                    if next_sequence:
                        continue
                    epoch_images = np.concatenate(epoch_images, axis = 2)
                    sequences.append(epoch_images[np.newaxis,...])
                sequences = np.concatenate(sequences, axis = 0)
                return sequences, exceptions

            noisy_sequences = []
            noisy_ids = []
            noisy_mjd = []
            modes = ["gaussian", "poisson"]
            sigmas = [0.01, 0.05, 0.1]
            for sigma in sigmas:
                for mode in modes:
                    ns_seq, exceptions = add_noise(sigma**2, mode)
                    noisy_sequences.append(ns_seq)
                    noisy_mjd.append(np.delete(self.sn_mjd_obs, exceptions, 0))
                    for i, sn_id in enumerate(self.sn_id):
                        if i in exceptions:
                            continue
                        noisy_ids.append(sn_id + "_ns" + mode +"_" + str(sigma))
            print(np.amax(self.sn_sequences))
            print(np.amax(np.concatenate(noisy_sequences, axis = 0)))

            self.sn_sequences = np.concatenate([self.sn_sequences, np.concatenate(noisy_sequences, axis = 0)], axis = 0)
            self.sn_id = self.sn_id + noisy_ids
            self.sn_mjd_obs = np.concatenate([self.sn_mjd_obs, np.concatenate(noisy_mjd, axis = 0)], axis = 0)
            print("shapes after adding noise")
            print("sequences shape = " + str(self.sn_sequences.shape))
            print("sn_mjd_obs shape = " + str(self.sn_mjd_obs.shape))
            print("len sn_id = " + str(len(self.sn_id)))

        blur_sequences()
        rotate_sequences()
        flip_sequences()
        add_noise_to_sequences()

        with open(self.CCDs_directory + 'sn_augmentation.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump({"data": self.sn_sequences, "mjd-obs": self.sn_mjd_obs, "id": self.sn_id}, f)

class Catalog(object):

    def __init__(self, image_size, extraction_method):
        print("Loading catalog: "+extraction_method)
        self.edge_size = 5 #in pixels
        self.stride = 21
        self.image_size = 21
        self.extraction_method = extraction_method
        if extraction_method == 'stride':
            self.get_stride_catalog(image_size)
        elif extraction_method == 'noise':
            self.get_noise_level_catalog(image_size)
        elif extraction_method == 'sextractor':
            self.get_hits_catalog()


    def get_stride_catalog(self, image_size):
        self.image_shape = image_size
        half_image = int(np.floor(self.image_size/2))
        x_limits = [half_image + self.edge_size, self.image_shape[0] - self.edge_size - half_image]
        y_limits = [half_image + self.edge_size, self.image_shape[1] - self.edge_size - half_image]
        x_values = np.arange(x_limits[0], x_limits[1], self.stride)
        y_values = np.arange(y_limits[0], y_limits[1], self.stride)
        #Catalog with object coordinates
        self.coordinate_list = [np.array([x,y]) for x in x_values for y in y_values]
        self.n_objects = len(self.coordinate_list)

    def get_ccd_coordinates(self, CCD):
        if self.extraction_method == "stride":
            return self.coordinate_list
        elif self.extraction_method == "sextractor":
            return self.ccd_catalog[CCD]

    def collapse_epoch_catalog(self):
        #TODO: Get joint of catalogs on each epoch
        self.ccd_catalog = {}
        count = 1
        for key, value in self.catalog_dict.iteritems():
            print("collapsing "+key)
            print(str(count)+"/87")
            start = time.time()
            repeated_objects = []
            object_catalog = value["coordinates"][0]
            object_epoch = value["coordinates"]
            for object_list in object_epoch:
                #print(object_catalog)
                #print(object_list)
                #asdasdasd
                #object_list = object_epoch[1]
                for i, object1 in enumerate(object_catalog):
                    #min_distance = 2000
                    for j, object2 in enumerate(object_list):
                        distance = norm(object1 - object2)
                        #print(distance)
                        if distance <= 2.5:
                            repeated_objects.append(j)
                            break
                        #if distance<=min_distance:
                        #    min_distance = distance
                    #print min_distance
                
                new_objects = list(set(range(len(object_list))) - set(repeated_objects))
                #print(new_objects)
                object_catalog = np.concatenate([object_catalog,object_list[new_objects,:]], axis=0)
            count += 1
            self.ccd_catalog[key] = object_catalog
            end = time.time()
            print("Time: " + str((end-start)/60.0) +"min")
        

    def get_hits_catalog(self):
        CCDs_directory = '/home/toshiba/rodrigo/R2015CCDs/'
        self.catalog_dict = np.load(CCDs_directory + "sextractor_catalog.pkl")
        self.collapse_epoch_catalog()
        with open(CCDs_directory + 'finalCCDcatalog.pkl', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.ccd_catalog, f)


if __name__=='__main__':
    database = HiTSData()

