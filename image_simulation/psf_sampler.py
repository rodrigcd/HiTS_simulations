import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from tqdm import tqdm
#plt.switch_backend('agg')
import cv2

class PSFSampler(object):

    def __init__(self, **kwargs):
        #print("PSF Sampler")
        np.random.seed(0)
        self.cam_obs_cond_path = kwargs["camera_and_obs_cond_path"]
        self.augmented_psfs = kwargs["augmented_psfs"]
        self.psfs = np.load(self.cam_obs_cond_path)["psf"]
        self.nx, self.ny, self.n_psfs = self.psfs.shape
        if self.augmented_psfs:
            self.increase_psf_database()
        self.psf_match(plot_n_examples=0)


    def psf_match(self, plot_n_examples=20):
        print("Doing gaussian fit to psf to compute SEEING")

        n_plotted = 0
        x, y = np.mgrid[-self.nx / 2.: self.nx / 2., -self.ny / 2.: self.ny / 2.]

        fit_p = fitting.LevMarLSQFitter()

        self.psf_seeing_xy = [] # [x, y]EB_set_good_EB_2500.hdf5
        for i in range(self.psfs.shape[2]):
            p_init = models.Gaussian2D(amplitude=np.mean(self.psfs[..., i]), x_mean=0.0, y_mean=0,
                                       #x_stddev=1.7, y_stddev=1.7)
                                        cov_matrix=np.array([[1.7, 0], [0, 1.7]])
                                       )
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                p = fit_p(p_init, x, y, self.psfs[..., i])
                x_std, y_std, theta = p.x_stddev.value, p.y_stddev.value, p.theta.value
                #cov = p.cov_matrix.value
                x_seeing, y_seeing = 2.0*np.sqrt(2.0*np.log(2.0))*np.array([x_std, y_std])
                fwhm_x, fwhm_y = p.x_fwhm, p.y_fwhm
                #print(x_seeing, y_seeing)
                #print(fwhm_x, fwhm_y)
                #if i == 10:
                #    asdasd
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
        diff_seeing = np.abs(self.psf_seeing - seeing - np.random.uniform(low=-0.3, high=0.3))
        best_seeing_index = np.argmin(diff_seeing)
        best_seeing_match = self.psf_seeing[best_seeing_index]
        #best_seeing_index += np.random.randint(low=-1, high=2) #randomizing a little
        if best_seeing_index >= self.ordered_psfs.shape[-1]:
            best_seeing_index = self.ordered_psfs.shape[-1]-1
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

    def augment_psf(self, psf_image, angle=np.pi/6.0, shrink=0.5, reduce=0.1):
        rows,cols = psf_image.shape
        # Rotation Matrix
        rm = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center = np.zeros(shape=(1,2))
        p1 = np.array([3, 0])[..., np.newaxis]
        p2 = np.array([0, 3])[..., np.newaxis]
        # Shrink parameter
        shrink_vector = np.array([-shrink, shrink])[..., np.newaxis]
        shrink_vector = np.dot(rm, shrink_vector).T
        p1_f = np.dot(rm, p1-p1*reduce).T# + shrink_vector).T
        p2_f = np.dot(rm, p2-p2*reduce).T# + np.flip(shrink_vector, axis=0)).T
        pts1 = np.concatenate([center, p1.T, p2.T]).astype(np.float32) +10
        pts2 = np.concatenate([center,
                               p1_f+shrink_vector,
                               p2_f+np.flip(shrink_vector, axis=1)]).astype(np.float32) +10

        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(psf_image,M,(cols,rows))
        dst = dst/np.sum(dst)
        return dst

    def increase_psf_database(self, n_times = 4):
        augmented_psfs = []
        for i in range(n_times):
            angles = np.random.uniform(low=0, high=np.pi*2, size=(self.psfs.shape[2],))
            shrinks = np.random.uniform(low=0, high=0.55, size=(self.psfs.shape[2],))
            reduce = np.random.uniform(low=-0.1, high=0.25, size=(self.psfs.shape[2],))
            for psf_i in range(self.psfs.shape[2]):
                new_psf = self.augment_psf(self.psfs[..., psf_i],
                                           angle=angles[psf_i],
                                           shrink=shrinks[psf_i],
                                           reduce=reduce[psf_i])
                augmented_psfs.append(new_psf)
        augmented_psfs = np.stack(augmented_psfs, axis=2)
        self.psfs = np.concatenate([self.psfs, augmented_psfs], axis=2)
        print(self.psfs.shape)

if __name__ == "__main__":

    obs_cond_path = "../real_obs/pickles/camera_and_obs_cond.pkl"
    sampler = PSFSampler(camera_and_obs_cond_path=obs_cond_path)

    test_seeings = [4.13, 4.98, 5.373, 4.5, 4.42, 5.01]
    for i, seeing in enumerate(test_seeings):
        psf, real_s = sampler.sample_psf(seeing)
    #     plt.imshow(psf)
    #     plt.title("real_s: "+str(real_s)+", requested s: "+str(seeing))
    #     plt.savefig("plots/psf_match/sample_example"+str(i)+".png")
    #     plt.close("all")
