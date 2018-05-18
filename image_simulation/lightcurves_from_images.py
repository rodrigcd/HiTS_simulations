import h5py
import numpy as np
import time
import astropy.modeling.models as models
import itertools
import matplotlib.pyplot as plt

# TODO: Order by field and get the photometry using psf from images
class ImagePhotometry(object):

    def __init__(self, **kwargs):
        self.image_path = kwargs["images_path"]
        self.obs_condition_path = kwargs["obs_cond"]
        self.bands = kwargs["bands"]
        self.save_path = kwargs["save_path"]
        self.chunk_size = kwargs["chunk_size"]

        self.image_data = h5py.File(self.image_path, "r")
        self.fields = list(self.image_data.keys())
        self.cam_params = np.load(self.obs_condition_path)["camera_params"]

        self.average_camera_params()
        self.estimate_sky_from_images()
    def get_lightcurve(self, image_seq, gal_seq, psf, mask, sky, sky_variance):
        """
        :param image_seq: (21, 21, time)
        :param gal_seq:  (21, 21, time)
        :param psf:  (21, 21, time)
        :param mask:  (21, 21, time) boolean
        :param sky:  (time)
        :return:
        """
        estimated_lightcurve = []
        variance = []
        clean_source_values = []
        residuals = np.zeros(21, 21, len(sky))
        clean_image = image_seq - gal_seq - sky
        for time_index in range(self.clean_image.shape[0]):
            current_source_image = clean_image[:, :, time_index]
            current_psf = psf[..., time_index]
            current_mask = mask[..., time_index]
            current_psf = np.multiply(current_psf, current_mask)
            V_ij = sky_variance[time_index] \
                   + np.abs(current_source_image/np.sqrt(self.gain)) \
                   + gal_seq[:, :, time_index]/np.sqrt(self.gain)
            weights = np.divide(current_psf, V_ij) / np.sum(np.divide(np.square(current_psf), V_ij))
            variance.append(np.sum(np.multiply(np.square(weights), V_ij)))
            clean_source = np.multiply(weights, current_source_image)
            clean_source_values.append(clean_source)
            photometry_counts = np.sum(clean_source)
            estimated_lightcurve.append(np.sum(clean_source_values))
            residuals[:, :, time_index] = current_source_image - current_psf * photometry_counts

        return np.array(estimated_lightcurve), np.array(variance), residuals

    def average_camera_params(self):
        self.readout_noise = 0
        self.gain = 0
        n_cams = len(self.cam_params)
        for cam, params in self.cam_params.items():
            self.readout_noise += params["read_noise"]
            self.gain += params["gain"]
        self.readout_noise = self.readout_noise/np.float(n_cams)
        self.gain = self.gain/np.float(n_cams)

    def estimate_sky_from_images(self, field):
        field_data = self.image_data[field]
        field_images = field_data["images"]
        sky_from_data = field_data["obs_cond"]["sky_brightness"]["g"][:]
        estimated_sky_from_images = []
        estimated_std = []
        for time in range(sky_from_data.shape[0]):
            time_sky = 0
            pixel_array = []
            for i in range(1000):
                pixel_array.append(field_images["g"][i, :, 0, time])
            pixel_array = np.concatenate(pixel_array, axis=0)
            print(pixel_array.shape)
            time_sky = np.mean(pixel_array)
            estimated_sky_from_images.append(time_sky)
            estimated_std.append(np.std(pixel_array, ddof=1))
        # print(sky_from_data, estimated_sky_from_images)
        # print(sky_from_data/np.sqrt(self.gain), np.array(estimated_std)**2)
        return np.array(estimated_sky_from_images), np.array(estimated_std)**2

    def run_photometry(self):
        #TODO: ALMOST DONE
        return

if __name__ == "__main__":
    image_path = "/home/rcarrasco/simulated_data/image_sequences/may16_fixed_eb_psf2500.hdf5"
    camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
    save_path = "/home/rcarrasco/simulated_data/image_sequences/lightcurves_from_images/"
    file_name = "may16"
    bands = ["g",]
    chunk_size = 100

    photometry = ImagePhotometry(images_path=image_path,
                                 obs_cond=camera_and_obs_cond_path,
                                 bands=bands,
                                 save_path=save_path,
                                 chunk_size=chunk_size)