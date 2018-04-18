import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import itertools
import pprint
from astropy.modeling.functional_models import Sersic2D
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from MagCounts import Mag2Counts


class GalaxyImages(object):
    """Generates galaxy images"""
    def __init__(self, **kwargs):
        self.distr_path = kwargs["distr_path"]
        self.stamp_size = kwargs["stamp_size"]
        self.image_size = kwargs["image_size"]
        self.pixel_size = kwargs["pixel_size"]
        self.n_integrations = kwargs["n_integrations"]
        self.bands = kwargs["bands"]
        self.load_distr = kwargs["load_all_data"]

        if self.load_distr:
            self.distr_dataframe = pd.read_csv(self.distr_path)
            # Fixing Data
            self.fix_data()
            self.distr_features = [str(index) for index in self.distr_dataframe.columns]
        self.current_parameters = {}
        self.image_parameters = {}
        self.current_image = {}
        self.image_index_array = np.array(list(itertools.product(np.arange(self.image_size[0]),
                                                                 np.arange(self.image_size[1]))))
        self.image_distr_x = np.arange(len(self.image_index_array))
        self.galaxy_id = 0
        self.profile = {}
        # print("galaxy_params")
        # print(str(list(self.distr_dataframe)))
        # print(len(list(self.distr_dataframe)))
        # self.sample_galaxy()

    def fix_data(self):
        condition = (self.distr_dataframe["expRad_"+self.bands[0]] != 0.0) & \
                    (self.distr_dataframe["deVRad_"+self.bands[0]] != 0.0) & \
                    (self.distr_dataframe["deVAB_"+self.bands[0]] != 0.05) & \
                    (self.distr_dataframe["expAB_"+self.bands[0]] != 0.05)

        if len(self.bands) >= 2:
            for i in range(len(self.bands)-1):
                condition = condition & (self.distr_dataframe["expRad_"+self.bands[i+1]] != 0.0) & \
                    (self.distr_dataframe["deVRad_"+self.bands[i+1]] != 0.0) & \
                    (self.distr_dataframe["deVAB_"+self.bands[i+1]] != 0.05) & \
                    (self.distr_dataframe["expAB_"+self.bands[i+1]] != 0.05)

        self.distr_dataframe = self.distr_dataframe[condition]

    def sample_galaxy(self, redshift=[], z_tolerance=0.1, by_id=False, centered=False):
        self.redshift = redshift
        if not redshift:
            z_dataframe = self.distr_dataframe
        else:
            z_dataframe = self.distr_dataframe[(self.distr_dataframe["z"] > redshift-z_tolerance) &
                                               (self.distr_dataframe["z"] < redshift+z_tolerance)]
            if z_dataframe.empty:
                print("no redshift match")
                z_dataframe = self.distr_dataframe
        if by_id:
            self.current_parameters = z_dataframe.iloc[self.galaxy_id, :]
            self.galaxy_id += 1
            if self.galaxy_id >= len(z_dataframe):
                print("All galaxies sampled")
        else:
            self.current_parameters = z_dataframe.sample(n=1).iloc[0]
        self.position_error = np.random.normal(scale=0.3, size=2)
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.current_parameters["angle"] = self.angle
        self.create_profile(pixel_size=self.pixel_size)
        if centered:
            self.pos = np.array([[self.image_size[0]/2, self.image_size[1]/2], ])
        else:
            self.pos = self.sample_position_galaxy()
        self.current_parameters["pos_row"] = self.pos[0, 0]
        self.current_parameters["pos_col"] = self.pos[0, 1]

        return self.current_parameters

    def create_profile(self, pixel_size, params={}):
        if len(params) == 0:
            params = self.current_parameters
        band_profile = {}
        for band in self.bands:
            # devacouler profile
            ellip_de = 1 - params["deVAB_" + band]
            r_eff_de = params["deVRad_" + band] / pixel_size
            prop_dev = params["fracDeV_" + band]
            # exp profile
            ellip_exp = 1 - params["expAB_" + band]
            r_eff_exp = params["expRad_" + band] / pixel_size
            # making profile
            x, y = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]))
            band_profile["mod1"] = Sersic2D(amplitude=1, r_eff=r_eff_exp, n=1, x_0=self.image_size[0] / 2,
                                            y_0=self.image_size[1] / 2, ellip=ellip_exp, theta=params["angle"])
            band_profile["mod2"] = Sersic2D(amplitude=prop_dev, r_eff=r_eff_de, n=4, x_0=self.image_size[0] / 2,
                                            y_0=self.image_size[1] / 2, ellip=ellip_de, theta=params["angle"])
            #self.img1, self.img2 = self.mod1(x, y), self.mod2(x, y)
            band_profile["img2"] = self.random_integration(prop_dev=prop_dev, r_eff=r_eff_de,
                                                           ellip=ellip_de, angle=params["angle"])
            band_profile["img1"] = band_profile["mod1"](x, y)
            band_profile["norm_image"] = (band_profile["img1"] + band_profile["img2"]) /\
                                          np.sum(band_profile["img1"] + band_profile["img2"])
            band_profile["disc_image"] = band_profile["img1"]/np.sum(band_profile["img1"])
            self.profile[band] = band_profile
            band_profile = {}

    def random_integration(self, prop_dev, r_eff, ellip, angle):
        """random integration of bulge"""
        n_iter = self.n_integrations
        x, y = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]))
        images = []
        for i in range(n_iter):
            noise = np.random.uniform(low=0, high=0.8, size=2)
            mod = Sersic2D(amplitude=prop_dev, r_eff=r_eff, n=4, x_0=self.image_size[0] / 2+noise[0],
                           y_0=self.image_size[1] / 2+noise[1], ellip=ellip, theta=angle)
            images.append(mod(x, y)[np.newaxis, ...])
        images = np.concatenate(images, axis=0)
        img = np.sum(images, axis=0)/n_iter
        return img

    def create_galaxy_image(self, band, t_exp, seeing, airmass, zero_point, airmass_term, params={}):
        if len(params) == 0:
            params = self.current_parameters

        #for i, band in enumerate(self.bands):
        mag = params["petroMag_" + band]
        galaxy_counts = Mag2Counts(mag, airmass, t_exp, zero_point, airmass_term=airmass_term)
        img = self.profile[band]["norm_image"] * galaxy_counts
        kernel = Gaussian2DKernel(seeing / (2 * np.sqrt(2)))
        img = convolve(img, kernel)
        self.current_image[band] = img
        return self.current_image, self.profile

    def sample_position_galaxy(self, n_samples=1, profile="disc"):
        if profile == "disc":
            image_to_sample = self.profile[self.bands[0]]["disc_image"]
        else:
            image_to_sample = self.profile[self.bands[0]]["norm_image"]
        if not np.isclose(np.sum(image_to_sample), 1):
            return True, image_to_sample, self.current_parameters
            # image_to_sample = image_to_sample/np.sum(image_to_sample)
        reshaped_image = np.reshape(image_to_sample, newshape=(-1,))
        image_distr = stats.rv_discrete(name='image_distr', values=(self.image_distr_x, reshaped_image))
        index_sampled = image_distr.rvs(size=n_samples)
        return self.image_index_array[index_sampled]

    def generate_galaxy_stamp(self, band, t_exp, seeing, airmass, zero_point, airmass_term, params={}):
        if len(params) == 0:
            params = self.current_parameters
        img, _ = self.create_galaxy_image(band, t_exp, seeing, airmass, zero_point, airmass_term, params)
        row_limits = np.array([params["pos_row"]-self.stamp_size[0]/2,
                               params["pos_row"]+self.stamp_size[0]/2+1], dtype=np.int)
        column_limits = np.array([params["pos_col"]-self.stamp_size[1]/2,
                                  params["pos_col"]+self.stamp_size[1]/2+1], dtype=np.int)
        if row_limits[0] < 0:
            row_limits -= row_limits[0]
        elif row_limits[1] >= self.image_size[0]:
            row_limits -= row_limits[1] - self.image_size[0]
        if column_limits[0] < 0:
            column_limits -= column_limits[0]
        elif column_limits[1] >= self.image_size[1]:
            column_limits -= column_limits[1] - self.image_size[1]
        stamp = img[band][row_limits[0]:row_limits[1], column_limits[0]:column_limits[1]]
        return stamp

    def return_params(self):
        return self.current_parameters

    def return_headers(self):
        aux_frame = self.return_params()
        headers = aux_frame.keys()
        return headers


if __name__ == "__main__":
    distr_path = "/home/toshiba/rodrigo/galaxies/gal_mags_dev_exp_z_all_Filter_rodrigocd.csv"
    pixel_size = 0.4
    stamp_size = (21, 21)
    image_size = (60, 60)
    galaxies_gen = GalaxyImages(distr_path=distr_path,
                                pixel_size=pixel_size,
                                stamp_size=stamp_size,
                                image_size=image_size,
                                n_integrations=10,
                                bands=["g", "i", "r", "z"])

    t_exp = 225.0
    seeing = 4.3
    airmass = 1.1
    zero_point = 26.59

    frame = galaxies_gen.distr_dataframe
    print(frame.shape)
    print(frame[frame["expRad_z"] == 0].shape)

    redshift = 0.6
    galaxies_gen.sample_galaxy(redshift)
    galaxies_gen.generate_galaxy_stamp("g", t_exp, seeing, airmass, zero_point)
