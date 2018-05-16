import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from tqdm import tqdm
plt.switch_backend('agg')

class PSFSampler(object):

    def __init__(self, **kwargs):
        #print("PSF Sampler")
        self.cam_obs_cond_path = kwargs["camera_and_obs_cond_path"]
        self.psfs = np.load(self.cam_obs_cond_path)["psf"]
        self.nx, self.ny, self.n_psfs = self.psfs.shape
        self.psf_match()

    def psf_match(self, plot_n_examples=20):
        print("Doing gaussian fit to psf to compute SEEING")

        n_plotted = 0
        x, y = np.mgrid[-self.nx / 2.: self.nx / 2., -self.ny / 2.: self.ny / 2.]

        fit_p = fitting.LevMarLSQFitter()

        self.psf_seeing_xy = [] # [x, y]EB_set_good_EB_2500.hdf5
        for i in range(self.psfs.shape[2]):
            p_init = models.Gaussian2D(amplitude=np.mean(self.psfs[..., i]), x_mean=0.0, y_mean=0,
                                       x_stddev=1.9, y_stddev=1.9)

            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                p = fit_p(p_init, x, y, self.psfs[..., i])
                x_std, y_std = p.x_stddev.value, p.y_stddev.value
                x_seeing, y_seeing = 2.0*np.sqrt(2.0*np.log(2.0))*np.array([x_std, y_std])
                self.psf_seeing_xy.append(np.array([x_seeing, y_seeing]))

            if n_plotted < plot_n_examples:
                plt.figure(figsize=(8, 2.5))
                plt.subplot(1, 3, 1)
                plt.imshow(self.psfs[..., i], origin='lower', interpolation='nearest')
                plt.colorbar()
                plt.title("Data")
                plt.subplot(1, 3, 2)
                plt.imshow(p(x, y), origin='lower', interpolation='nearest')
                plt.colorbar()
                plt.title("Model seeing: "+str(self.psf_seeing_xy[-1]))
                plt.subplot(1, 3, 3)
                plt.imshow(self.psfs[..., i] - p(x, y), origin='lower', interpolation='nearest')
                plt.colorbar()
                plt.title("Residual")
                plt.savefig("plots/psf_match/psf_match_"+str(i)+".png")
                plt.close("all")
                n_plotted += 1
                #plt.show()
        self.psf_seeing_xy = np.stack(self.psf_seeing_xy, axis=0)
        self.psf_seeing = np.mean(self.psf_seeing_xy, axis=1)
        ordered_index = np.argsort(self.psf_seeing)
        self.psf_seeing = self.psf_seeing[ordered_index]
        self.ordered_psfs = self.psfs[..., ordered_index]
        bins = np.linspace(3, 10, 70)
        h_seeing, _ = np.histogram(self.psf_seeing, bins=bins, density=True)
        plt.figure(figsize=(12, 7))
        plt.plot(bins[1:], h_seeing)
        plt.title("Density of seeing in psfs")
        plt.xlabel("SEEING")
        plt.savefig("plots/psf_match/seeing_distribution.png")
        plt.close("all")

    def sample_psf(self, seeing, random_rotation=True, random_mirror=True):
        diff_seeing = np.abs(self.psf_seeing - seeing)
        best_seeing_index = np.argmin(diff_seeing)
        best_seeing_match = self.psf_seeing[best_seeing_index]
        best_seeing_index += np.random.randint(low=-1, high=2) #randomizing a little
        if best_seeing_index >= self.ordered_psfs.shape[1]:
            best_seeing_index = self.ordered_psfs.shape[1]-1
        elif best_seeing_index < 0:
            best_seeing_index = 0
        best_psf = self.ordered_psfs[..., best_seeing_index]
        psf_to_return = np.copy(best_psf)

        if random_rotation:
            k = np.random.randint(low=4, size=1)
            psf_to_return = np.rot90(psf_to_return, k=k)

        if random_mirror:
            k = np.random.randint(low=3, size=1)
            if k == 1:
                psf_to_return = np.fliplr(psf_to_return)
            elif k == 2:
                psf_to_return = np.flipud(psf_to_return)
        return psf_to_return, best_seeing_match

if __name__ == "__main__":

    obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
    sampler = PSFSampler(camera_and_obs_cond_path=obs_cond_path)

    test_seeings = [4.13, 4.98, 5.373, 4.5, 4.42, 5.01]
    for i, seeing in enumerate(test_seeings):
        psf, real_s = sampler.sample_psf(seeing)

        plt.imshow(psf)
        plt.title("real_s: "+str(real_s)+", requested s: "+str(seeing))
        plt.savefig("plots/psf_match/sample_example"+str(i)+".png")
        plt.close("all")
