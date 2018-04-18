import astropy.modeling.models as models
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from Galaxies import GalaxyImages


class ImageFactory:

    def __init__(self, **kwargs):
        self.nx, self.ny = kwargs["nx"], kwargs["ny"]
        self.readout_noise = kwargs["readout_noise"]
        self.x, self.y = np.mgrid[-self.nx/2.: self.nx/2., -self.ny/2.: self.ny/2.] + 0.5
        self.gain = kwargs["gain"]
        self.pixel_saturation = kwargs["pixel_saturation"]
        self.astrometric_error = kwargs["astrometric_error"]
        self.pixel_size = kwargs["pixel_size"]
        self.galaxies_distr_path = kwargs["galaxies_distr_path"]
        self.zero_points = kwargs["zero_point"] #["g", "r", "i"]
        self.exp_times = kwargs["exp_times"]
        self.airmass_terms = kwargs["airmass_terms"]
        self.bands = kwargs["bands"]
        self.sky_clipping = kwargs["sky_clipping"]

        self.galaxies_gen = GalaxyImages(distr_path=self.galaxies_distr_path,
                                         pixel_size=self.pixel_size,
                                         stamp_size=(self.nx, self.ny),
                                         image_size=(3*self.nx, 3*self.ny),
                                         zero_points=self.zero_points,
                                         n_integrations=20,
                                         bands=self.bands,
                                         load_all_data=True)

    def createPSFImage(self, band, counts, seeing, airmass, sky_counts, mean, zero_point, exp_time):
        """each input is just a scalar"""
        # seeing to sigmas
        sigma = seeing/(2*np.sqrt(2*np.log(2)))
        # gaussian model

        model = models.Gaussian2D(x_mean=mean[0], y_mean=mean[1],
                                  x_stddev=sigma, y_stddev=sigma)
        data = model(self.x, self.y)
        # insert gaussian
        # data = np.round(data*counts/data.sum()).astype(int)
        data = np.floor(data*counts/np.sum(data))
        # print("counts: "+ str(counts))
        # print("integral of psf: " + str(np.sum(data)))
        # estimated_sky_counts = 4*np.pi*(sigma**2)*sky_counts
        data += np.floor(np.amin([sky_counts, self.sky_clipping]))
        # adding galaxy
        if self.with_galaxy:
            galaxy_image = self.galaxies_gen.generate_galaxy_stamp(band, exp_time, seeing, airmass,
                                                            zero_point, airmass_term=self.airmass_terms[band])
            data += galaxy_image
        else:
            galaxy_image = np.zeros(shape=(self.nx, self.ny))
        # from counts to e
        data = data*self.gain
        # add Poisson noise e-
        data = np.random.poisson(data)

        # add readout noise e- (Gaussian)
        data += np.random.normal(scale=self.readout_noise, size=data.shape).astype(int)
        # from e- to counts
        data = data/self.gain
        data = np.clip(data, 0, self.pixel_saturation)
        data = np.floor(data)
        data.astype(int)
        return data, galaxy_image

    def createLightCurveImages(self, counts, seeing, airmass, sky_counts, zero_point, with_galaxy, redshift=[]):
        """ Creates light curve image from counts, each variable (counts, seeing, airmass, sky_counts)
        should be a dictionary with keys for each band"""
        images = {}
        gal_image = {}
        cov = np.diag([self.astrometric_error**2, self.astrometric_error**2])
        self.with_galaxy = with_galaxy
        if with_galaxy:
            self.galaxies_gen.sample_galaxy(redshift=redshift)
        for band in self.bands:
            x_mean, y_mean = np.random.multivariate_normal([0, 0], cov, counts[band].shape[1]).T
            band_image = np.zeros((self.nx, self.ny, counts[band].shape[1]), dtype="int")
            band_gal_image = np.zeros((self.nx, self.ny, counts[band].shape[1]), dtype="int")
            for i in range(counts[band].shape[1]):
                mean = [x_mean[i], y_mean[i]]
                im, gal = self.createPSFImage(band, counts[band][0, i], seeing[band][i],
                                              airmass[band][i], sky_counts[band][i], mean, zero_point[band],
                                              self.exp_times[band])
                band_image[..., i] = im[:]
                band_gal_image[..., i] = gal[:]
            images[band] = band_image
            gal_image[band] = band_gal_image
        return images, gal_image


if __name__ == "__main__":
    gain = 1.67 #[e-/ADU]
    counts = [2000, 3000] #ADU
    sky_counts = [10, 30] #ADU/pixel
    seeing = 5 #pixel
    nx, ny = 21, 21 #pixels
    readout_noise = 5 #sigma of gaussian (ADU)
    imfact = ImageFactory(nx, ny, readout_noise, gain)
    images, counts = imfact.createLightCurveImages(counts, seeing, sky_counts)
    print(images.shape)
    plt.imshow(images[:, :, 0])
    plt.savefig("image1.png")
    plt.imshow(images[:, :, 1])
    plt.savefig("image2.png")








