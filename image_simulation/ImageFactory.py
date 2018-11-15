import astropy.modeling.models as models
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from Galaxies import GalaxyImages
from psf_sampler import PSFSampler

class ImageFactory:

    def __init__(self, **kwargs):
        # print("Image Factory")
        self.nx, self.ny = kwargs["nx"], kwargs["ny"]
        self.x, self.y = np.mgrid[-self.nx/2.: self.nx/2., -self.ny/2.: self.ny/2.] + 0.5
        self.astrometric_error = kwargs["astrometric_error"]
        self.bands = kwargs["bands"]
        self.sky_clipping = kwargs["sky_clipping"]
        self.galaxies_distr_path = kwargs["galaxies_distr_path"]
        self.real_psfs = kwargs["real_psfs"]
        if self.real_psfs:
            print("- Using PSF sampler")
            self.psf_sampler = PSFSampler(camera_and_obs_cond_path=kwargs["obs_cond_path"],
                                          augmented_psfs=kwargs["augmented_psfs"])
        else:
            self.airmass_terms = kwargs["airmass_term"]

        self.ccd_parameters = kwargs["ccd_parameters"]
        self.readout_noise = self.ccd_parameters["read_noise"]
        self.gain = self.ccd_parameters["gain"]
        self.pixel_saturation = self.ccd_parameters["saturation"]
        self.pixel_size = self.ccd_parameters["pixel_scale"]

        #self.zero_points = kwargs["zero_point"] #["g", "r", "i"]
        #self.exp_times = kwargs["exp_times"]
        #self.airmass_terms = 0.15 #HardCoded

        print("- Galaxy images")
        self.galaxies_gen = GalaxyImages(distr_path=self.galaxies_distr_path,
                                         pixel_size=self.pixel_size,
                                         stamp_size=(self.nx, self.ny),
                                         image_size=(3*self.nx, 3*self.ny),
                                         n_integrations=20,
                                         bands=self.bands,
                                         load_all_data=True)

    def set_ccd_params(self, params):
        self.ccd_parameters = params
        self.readout_noise = self.ccd_parameters["read_noise"]
        self.gain = self.ccd_parameters["gain"]
        self.pixel_saturation = self.ccd_parameters["saturation"]
        self.pixel_size = self.ccd_parameters["pixel_scale"]

    def createPSFImage(self, band, counts, seeing, airmass, sky_counts, mean, zero_point, exp_time):
        """each input is just a scalar"""
        if self.real_psfs:
            psf, _ = self.psf_sampler.sample_psf(seeing)
            #print(seeing)
            #print(psf)
            data = np.copy(psf)
        else:
            # seeing to sigmas
            sigma = seeing/(2*np.sqrt(2*np.log(2)))
            # gaussian model

            model = models.Gaussian2D(x_mean=mean[0], y_mean=mean[1],
                                      x_stddev=sigma, y_stddev=sigma)
            data = model(self.x, self.y)
            psf = np.copy(data)
        # insert gaussian
        # data = np.round(data*counts/data.sum()).astype(int)
        data = np.round(data*counts/np.sum(data))
        # print("counts: "+ str(counts))
        # print("integral of psf: " + str(np.sum(data)))
        # estimated_sky_counts = 4*np.pi*(sigma**2)*sky_counts
        data += np.round(np.amin([sky_counts, self.sky_clipping]))
        # adding galaxy
        if self.with_galaxy:
            if self.real_psfs:
                galaxy_image = self.galaxies_gen.generate_galaxy_stamp(band=band,
                                                                       t_exp=exp_time,
                                                                       seeing=seeing,
                                                                       airmass=None,
                                                                       zero_point=zero_point,
                                                                       airmass_term=0.15,
                                                                       psf=psf)
            else:
                galaxy_image = self.galaxies_gen.generate_galaxy_stamp(band, exp_time, seeing, airmass,
                                                                       zero_point,
                                                                       airmass_term=self.airmass_terms[band])
            data += galaxy_image
        else:
            galaxy_image = np.zeros(shape=(self.nx, self.ny))
        # from counts to e
        data = data*self.gain
        # add Poisson noise e-
        data = np.random.poisson(np.clip(data, a_min=0, a_max=None))

        # add readout noise e- (Gaussian)
        data += np.random.normal(scale=self.readout_noise, size=data.shape).astype(int)
        # from e- to counts
        data = data/self.gain
        data = np.clip(data, 0, self.pixel_saturation)
        data = np.round(data, decimals=0)
        data.astype(int)
        return data, galaxy_image, psf

    def createLightCurveImages(self, counts, seeing, airmass, sky_counts, exp_time, zero_point, with_galaxy, redshift=[]):
        """ Creates light curve image from counts, each variable (counts, seeing, airmass, sky_counts)
        should be a dictionary with keys for each band"""
        images = {}
        gal_image = {}
        psf_image = {}
        cov = np.diag([self.astrometric_error**2, self.astrometric_error**2])
        self.with_galaxy = with_galaxy
        if with_galaxy:
            self.galaxies_gen.sample_galaxy(redshift=redshift)

        for band in self.bands:
            # x_mean, y_mean = np.random.multivariate_normal([0, 0], cov, counts[band].shape[1]).T
            x_mean, y_mean = np.random.uniform(low=-0.5, high=0.5, size=(2, counts[band].shape[1]))
            band_image = np.zeros((self.nx, self.ny, counts[band].shape[1]), dtype="int")
            band_gal_image = np.zeros((self.nx, self.ny, counts[band].shape[1]), dtype="int")
            band_psf_image = np.zeros((self.nx, self.ny, counts[band].shape[1]), dtype="float32")
            for i in range(counts[band].shape[1]):
                mean = [x_mean[i], y_mean[i]]
                if airmass is None:
                    im, gal, psf = self.createPSFImage(band, counts[band][0, i], seeing[band][i],
                                                  airmass, sky_counts[band][i], mean, zero_point[band][i],
                                                  exp_time[band][i])
                else:
                    im, gal, psf = self.createPSFImage(band, counts[band][0, i], seeing[band][i],
                                                  airmass[band][i], sky_counts[band][i], mean, zero_point[band][i],
                                                  exp_time[band][i])
                band_image[..., i] = im[:]
                band_gal_image[..., i] = gal[:]
                band_psf_image[..., i] = psf[:]
            images[band] = band_image
            gal_image[band] = band_gal_image
            psf_image[band] = band_psf_image
        return images, gal_image, psf_image


if __name__ == "__main__":
    a = 2








