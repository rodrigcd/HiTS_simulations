import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
import matplotlib as mpl
from os import listdir
from os.path import isfile, join
import pickle

class MagnitudeDistribution(object):

    def __init__(self, **kwargs):
        self.load_distr = kwargs["load_distr"]
        self.bands = kwargs["bands"]
        self.extrapolation_limit = kwargs["extrapolation_limit"]

        # Methods in init
        if not self.load_distr:
            self.distr_path = kwargs["distribution_path"]
            self.fit_limits = kwargs["fit_limits"]
            self.mag_column = "MedianKronMag"
            self.file_list = [f for f in listdir(self.distr_path) if isfile(join(self.distr_path, f))]
            self.panda_frames = {}
            self.mag_values = {}
            self.distr_per_band = {}
            self.read_mag_file()
            self.extract_values()
            self.extend_distributions()
        else:
            self.load_coef()

    def read_mag_file(self):
        for file_name in self.file_list:
            if file_name[0] == "u":
                aux_frame = pandas.read_csv(self.distr_path+"/"+file_name)
                self.panda_frames["u"] = aux_frame[aux_frame["u"+self.mag_column] != 0]
            else:
                aux_frame = pandas.read_csv(self.distr_path+"/"+file_name)
                self.panda_frames["gri"] = aux_frame[(aux_frame["g" + self.mag_column] != 0) &
                                                     (aux_frame["r" + self.mag_column] != 0) &
                                                     (aux_frame["i" + self.mag_column] != 0)]

    def extract_values(self):
        for band in self.bands:
            if band == "u":
                self.mag_values[band] = self.panda_frames["u"]["u"+self.mag_column].values
            elif band in "gri":
                self.mag_values[band] = self.panda_frames["gri"][band+self.mag_column].values
            elif band == "z":  # Copying from i band
                # fake z distribution mag_i*1.07 - 2.0 just by visual inspection
                self.mag_values[band] = self.panda_frames["gri"]["i"+self.mag_column].values*1.07 - 2.0

    def extend_distributions(self):
        bins = np.linspace(12, 30, 100)
        self.coef_per_band = {}
        for band in self.bands:
            n, bins = np.histogram(self.mag_values[band], bins=bins, density=True)
            log_n = np.log(n)
            trunc_index = np.where(np.logical_and(bins > self.fit_limits[band][0],
                                                  bins < self.fit_limits[band][1]))[0]
            log_n = log_n[trunc_index]
            trunc_bins = np.log(bins[trunc_index])
            self.coef_per_band[band] = np.polyfit(trunc_bins, log_n, deg=1)
        with open('distr_coef.pkl', 'wb') as f:
            pickle.dump(self.coef_per_band, f)

    def load_coef(self):
        self.coef_per_band = np.load("distr_coef.pkl")

    def sample(self, n_samples):
        samples = {}
        for band in self.bands:
            mag_range = self.extrapolation_limit[band]
            coef = self.coef_per_band[band]
            x_range = np.linspace(mag_range[0], mag_range[1], 200)
            fn = np.vectorize(lambda x: x ** coef[0] * np.exp(coef[1]))
            Fn = np.vectorize(lambda x: np.exp(coef[1]) * (x ** (coef[0] + 1) / (coef[0] + 1)))
            Fn_inv = np.vectorize(lambda x: ((coef[0] + 1) / np.exp(coef[1]) * x) ** (1 / (coef[0] + 1)))

            multiplier = Fn(mag_range[1]) - Fn(mag_range[0])
            band_samples = Fn_inv(np.random.uniform(0, 1, size=n_samples) * multiplier + Fn(mag_range[0]))
            samples[band] = np.sort(band_samples)
        shuffled_index = np.arange(0, len(samples[self.bands[0]]), step=1)
        np.random.shuffle(shuffled_index)
        for band in self.bands:
            samples[band] = samples[band][shuffled_index]
        return samples


if __name__ == "__main__":

    # distr_path = "/home/rodrigo/supernovae_detection/object_distribution/"
    bands = ["g", "r", "i", "z"]
    extend_fit = 0
    load_distribution = True
    extrapolation_limits = {'g': [12, 25.602089154033994], 'r': [12, 25.029324900291915], 'i': [12, 24.45150161567846], 'z': [12, 23.122699702058064]}

    # fit_limits = {"z": [17.5, 20+extend_fit],
    #               "u": [16, 21+extend_fit],
    #               "g": [19, 22+extend_fit],
    #               "r": [18, 22+extend_fit],
    #               "i": [18, 20+extend_fit]}

    object_distribution = MagnitudeDistribution(# distribution_path=distr_path,
                                                extrapolation_limit=extrapolation_limits,
                                                # fit_limits=fit_limits,
                                                bands=bands,
                                                load_distr=load_distribution)

    bins = np.linspace(12, 30, 200)
    coefs = object_distribution.coef_per_band
    samples = object_distribution.sample(n_samples=10000)
    mpl.style.use("default")
    f, ax = plt.subplots(1, 1, figsize=(14, 7))
    # print(samples)
    for i, band in enumerate(bands):
        n_extrapolated, _ = np.histogram(samples[band], bins=bins, density=True)
        ax.plot(bins[1:], n_extrapolated, "oC"+str(i), label="extr "+band)
        n_estimated = np.log(bins[1:])*coefs[band][0] + coefs[band][1]
        #ax2.set_yscale("log")
    plt.xlabel("Magnitude", fontsize=16)
    ax.set_ylabel("Normalized Frequency", fontsize=16)
    ax.legend(fontsize=16)
    plt.savefig("custom_distr.png")
    plt.show()



