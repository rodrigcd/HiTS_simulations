import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
import numpy as np
import time
import astropy.modeling.models as models
import itertools
import matplotlib.pyplot as plt

class ImagePhotometry(object):

    def __init__(self, **kwargs):
        self.image_path = kwargs["images_path"]
        self.obs_condition_path = kwargs["obs_cond"]
        self.bands = kwargs["bands"]
        self.save_path = kwargs["save_path"]
        self.chunk_size = kwargs["chunk_size"]
        self.times_seeing = kwargs["times_seeing"]

        self.image_data = h5py.File(self.image_path, "r+")
        self.fields = list(self.image_data.keys())

        self.cam_params = np.load(self.obs_condition_path)["camera_params"]
        self.gruops_to_copy = ['count_lightcurves', 'galaxy_flag', 'ids',
                               'labels','lc_type', 'lightcurves', 'obs_cond']


        self.average_camera_params()
        #self.estimate_sky_from_images()


    def get_lightcurve(self, image_seq, gal_seq, psf, mask, sky, sky_variance, good_quality_points=None):
        """
        :param image_seq: (21, 21, time)
        :param gal_seq:  (21, 21, time)
        :param psf:  (21, 21, time)
        :param mask:  (21, 21, time) boolean
        :param sky:  (time)
        :return:
        """
        if good_quality_points is None:
            good_quality_points = np.ones(shape=sky.shape).astype(bool)

        estimated_lightcurve = []
        variance = []
        clean_source_values = []
        residuals = np.zeros(shape=(21, 21, len(sky)))
        clean_image = image_seq - gal_seq - sky
        valid_time_index = np.where(good_quality_points==True)[0]

        for time_index in valid_time_index:
            current_source_image = clean_image[:, :, time_index]
            current_psf = psf[..., time_index]
            current_mask = mask[..., time_index]
            masked_psf = np.multiply(current_psf, current_mask)
            V_ij = sky_variance[time_index] \
                   + np.abs(current_source_image/np.sqrt(self.gain)) \
                   + gal_seq[:, :, time_index]/np.sqrt(self.gain)
            weights = np.divide(current_psf, V_ij) / np.sum(np.divide(np.square(current_psf), V_ij))
            # variance.append(np.sum(np.multiply(np.square(weights), V_ij)))
            variance.append(np.sum(np.multiply(np.multiply(np.square(weights), V_ij), current_mask)))
            clean_source = np.multiply(np.multiply(weights, current_source_image), current_mask)
            clean_source_values.append(clean_source)
            photometry_counts = np.sum(clean_source)
            estimated_lightcurve.append(photometry_counts)
            residuals[:, :, time_index] = current_source_image - masked_psf * photometry_counts

        return np.array(estimated_lightcurve), np.array(variance), residuals, clean_image

    def average_camera_params(self):
        self.readout_noise = 0
        self.gain = 0
        n_cams = len(self.cam_params)
        for cam, params in self.cam_params.items():
            self.readout_noise += params["read_noise"]
            self.gain += params["gain"]
        self.readout_noise = self.readout_noise/np.float(n_cams)
        self.gain = self.gain/np.float(n_cams)

    def estimate_sky_from_images(self, field, n_images=1000):
        field_data = self.image_data[field]
        field_images = field_data["images"]
        sky_from_data = field_data["obs_cond"]["sky_brightness"]["g"][:]
        estimated_sky_from_images = []
        estimated_std = []
        for time in range(sky_from_data.shape[0]):
            time_sky = 0
            pixel_array = []
            for i in range(n_images):
                pixel_array.append(field_images["g"][i, :, 0, time])
            pixel_array = np.concatenate(pixel_array, axis=0)
            #print(pixel_array.shape)
            time_sky = np.mean(pixel_array)
            estimated_sky_from_images.append(time_sky)
            estimated_std.append(np.std(pixel_array, ddof=1))
        # print(sky_from_data, estimated_sky_from_images)
        # print(sky_from_data/np.sqrt(self.gain), np.array(estimated_std)**2)
        return np.array(estimated_sky_from_images), np.array(estimated_std)**2

    def get_apperture_mask(self, field_seeing, stamp_size=(21, 21), times_seeing=0.6731):
        aperture_mask_list = []
        aperture_radius_list = []
        for s in field_seeing:
            aperture_mask = np.zeros(shape=stamp_size)
            aperture_radius = self.times_seeing * s
            coordinates = np.array(list(itertools.product(range(stamp_size[0]), range(stamp_size[0]))))
            coordinates_polar = coordinates - np.array(stamp_size) / 2.0
            psf_mask = coordinates[np.linalg.norm(coordinates_polar, axis=1) <= aperture_radius]
            aperture_mask[psf_mask[:, 0], psf_mask[:, 1]] = 1
            aperture_mask_list.append(aperture_mask)
            aperture_radius_list.append(aperture_radius)
        return np.stack(aperture_mask_list, axis=2), aperture_radius_list

    def run_photometry(self, output_filename, band="g"):
        start = time.time()
        output_file = h5py.File(self.save_path + output_filename + ".hdf5", "w")
        for field in self.fields:
            print("Running "+field)
            field_group = output_file.create_group(name=field)

            image_field_group = self.image_data[field]
            if "estimated_counts" in list(image_field_group.keys()):
                del image_field_group["estimated_counts"]
            if "estimated_error_counts" in list(image_field_group.keys()):
                del image_field_group["estimated_error_counts"]
            count_group = image_field_group.create_group("estimated_counts")
            error_gruop = image_field_group.create_group("estimated_error_counts")

            image_field_data = self.image_data[field]
            for group_name in self.gruops_to_copy:
                image_field_data.copy(group_name, field_group)
                
            # Data for photometry
            images = image_field_data["images"][band][:]
            gal_images = image_field_data["galaxy_image"][band][:]
            psf_image = image_field_data["psf_image"][band][:]
            good_quality_points = image_field_data["obs_cond"]["good_quality_points"][band][:]
            sky_field, var_field = self.estimate_sky_from_images(field, n_images=300)
            sky_field = image_field_data["obs_cond"]["sky_brightness"][band][:]
            mask, _ = self.get_apperture_mask(field_group["obs_cond"]["seeing"][band][:])
            #print("images: "+str(images.shape))
            #print("gal: "+str(gal_images.shape))n
            #print("psf: "+str(psf_image.shape))
            #print(sky_field.shape, var_field.shape)
            #print(mask.shape)
            # yostart = time.time()

            estimated_lc = []
            estimated_variance = []
            residuals = []
            for i_image in range(images.shape[0]):
                est_lc, est_var, res, source_image = self.get_lightcurve(image_seq=images[i_image, ...],
                                                                         gal_seq=gal_images[i_image, ...],
                                                                         psf=psf_image[i_image, ...],
                                                                         mask=mask,
                                                                         sky=sky_field,
                                                                         sky_variance=var_field)
                estimated_lc.append(est_lc)
                estimated_variance.append(est_var)
                residuals.append(res)
            estimated_lc = np.stack(estimated_lc, axis=0)
            estimated_variance = np.stack(estimated_variance, axis=0)
            residuals = np.stack(residuals, axis=0)
            field_group.create_dataset("estimated_count_lc", data=estimated_lc)
            field_group.create_dataset("estimated_count_variance", data=estimated_variance)
            count_group.create_dataset(band, data=estimated_lc)
            error_gruop.create_dataset(band, data=estimated_variance)
            # field_group.create_dataset("residuals", data=residuals)
        end = time.time()
        print(file_name + " done in " + str(np.round(end-start, decimals=2)) + " sec")
        return

    def filter_by_conditions(self, output_filename, condition_limits):
        hdf5_file = h5py.File(self.save_path + output_filename + ".hdf5", 'r+')
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

if __name__ == "__main__":
    image_path = "/home/rcarrasco/simulated_data/image_sequences/complete_aug22_moredet2500.hdf5"
    #image_path = "/home/rcarrasco/simulated_data/image_sequences/psf_aug_july27_erf_distr2500.hdf5"
    #image_path = "/home/rcarrasco/simulated_data/image_sequences/complete_june8_erf_distr2500.hdf5"
    #image_path = "/home/rcarrasco/simulated_data/image_sequences/small_may30_erf_distr50.hdf5"
 
    camera_and_obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
    save_path = "/home/rcarrasco/simulated_data/image_sequences/lightcurves_from_images/"
    file_name = "more_detections"
    file_name = image_path.split("/")[-1].split(".")[0] + file_name
    bands = ["g",]
    chunk_size = 100
    times_seeing = 2.0*(1/(2*np.sqrt(2*np.log(2)))) # This is 2 sigma

    photometry = ImagePhotometry(images_path=image_path,
                                 obs_cond=camera_and_obs_cond_path,
                                 bands=bands,
                                 save_path=save_path,
                                 chunk_size=chunk_size,
                                 times_seeing=times_seeing)

    filter_by_conditions = {"seeing": {"g": [0, 2.0 / 0.27]}}  # fitler obs condition by range (seeing in pixels)

    photometry.run_photometry(output_filename=file_name, band="g")
    # photometry.filter_by_conditions(output_filename=file_name, condition_limits=filter_by_conditions)
