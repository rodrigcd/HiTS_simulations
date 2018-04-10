import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors.kde import KernelDensity
from gatspy.datasets import fetch_rrlyrae, RRLyraeGenerated
import warnings
from custom_distribution import MagnitudeDistribution
import matplotlib as mpl
warnings.filterwarnings("ignore")


class LightCurve(object):
    """Light curve main class"""
    def __init__(self, **kwargs):
        self.observation_days = kwargs["observation_days"]  # dictionary
        self.mag_generator = MagnitudeDistribution(**kwargs)
        self.bands = kwargs["bands"]

    def generate_lightcurves(self, n_lightcurves):
        """Returns a dictionary with keys per band with a list of light curves
        and dictionary with parameters of each light curve"""
        pass


class Asteroids(LightCurve):
    """Asteroid, params = magnitude"""
    def __init__(self, **kwargs):
        super(Asteroids, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves):
        lightcurves = {}
        mag_samples = self.mag_generator.sample(n_samples=n_lightcurves)
        time_samples = {}
        for band in self.bands:
            time_samples[band] = np.random.randint(low=0, high=len(self.observation_days[band]))
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(self.observation_days[band])))*40.0
            lightcurves[band][:, time_samples[band]] = mag_samples[band]
            mag_samples[band] = mag_samples[band].tolist()
        return lightcurves, mag_samples


class Constant(LightCurve):
    """Constant object light curve, params = magnitude"""
    def __init__(self,  mag_limit=25, **kwargs):
        super(Constant, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves):
        lightcurves = {}
        mag_samples = self.mag_generator.sample(n_samples=n_lightcurves)
        for band in self.bands:
            lightcurves[band] = np.repeat(mag_samples[band][..., np.newaxis],
                                          len(self.observation_days[band]),
                                          axis=1)
            # np.random.normal(loc=0, scale=0.01, size=lightcurves.shape) No noise
            mag_samples[band] = mag_samples[band].tolist()
        return lightcurves, mag_samples


class RRLyrae(LightCurve):
    """RRLyrae object light curve, params = mean(magnitude) sampled from distribution"""
    def __init__(self, obs_error=0.00, **kwargs):
        super(RRLyrae, self).__init__(**kwargs)
        self.obs_error = obs_error
        self.id_count = 0
        self.rrlyrae = fetch_rrlyrae()

    def generate_single_lc(self):
        lcid = self.rrlyrae.ids[self.id_count]
        period = self.rrlyrae.get_metadata(lcid)["P"]
        self.id_count += 1
        random_state_int = np.random.randint(low=0, high=10000, size=1)
        if self.id_count >= len(self.rrlyrae.ids):
            self.id_count = 0
        gen = RRLyraeGenerated(lcid, random_state=random_state_int)
        mag = {}
        for band in self.bands:
            mag[band] = gen.generated(band, self.observation_days[band], err=self.obs_error)
        return mag, period

    def generate_lightcurves(self, n_lightcurves, re_sampled=True):
        lightcurves = {}
        params = {}
        lightcurves_list = []
        mag_samples = self.mag_generator.sample(n_lightcurves)
        for i in range(n_lightcurves):
            lightcurves_list.append(self.generate_single_lc())
        for band in self.bands:
            lightcurves[band] = []
            params[band] = []
            for i in range(n_lightcurves):
                lc, period = lightcurves_list[i]
                if re_sampled:
                    lc[band] += -np.mean(lc[band]) + mag_samples[band][i]
                lightcurves[band].append(lc[band][np.newaxis, ...])
                params[band].append(np.array([mag_samples[band][i], period]))
            lightcurves[band] = np.concatenate(lightcurves[band], axis=0)
            params[band] = params[band]
        return lightcurves, params


class Supernovae(LightCurve):
    """RRLyrae object light curve, 
    params = 
    log(redshift)
    explosion time
    Av: atenuation in magnitude
    mass
    energy
    mdot: mass loss rate
    rcsm: maximum wind radius
    vwindf: terminal wind velocity
    beta: wind acceleration parameter"""
    def __init__(self, **kwargs):
        super(Supernovae, self).__init__(**kwargs)
        self.lc_path = kwargs["sn_lightcurves_path"]
        self.limmag = kwargs["limmag"]
        self.param_path = kwargs["sn_parameters_path"]

    def generate_lightcurves(self, n_lightcurves):
        filtered_lightcurves = {}
        parameters = {}
        all_lightcurves_data = np.load(self.lc_path)
        all_parameters_data = np.load(self.param_path)
        for band in self.bands:
            filtered_lightcurves[band] = []
            parameters[band] = []
        random_index = np.arange(len(all_lightcurves_data))
        np.random.shuffle(random_index)
        lc_count = 0
        for i in random_index:
            lc = all_lightcurves_data[i]
            current_lightcurve = lc
            detection_per_band = []
            for band in self.bands:
                mag_diff = self.limmag[band] - current_lightcurve[band]
                valid_index = np.where(mag_diff > 0)[0]
                detection_per_band.append(len(valid_index) > 0)
            if any(detection_per_band):
                lc_count += 1
                for band in self.bands:
                    filtered_lightcurves[band].append(lc[band][np.newaxis, :])
                    if len(all_parameters_data.shape) >= 2:
                        parameters[band].append(all_parameters_data[..., i])
                    else:
                        parameters[band].append(all_parameters_data[i])
            if n_lightcurves <= lc_count:
                break

        for band in self.bands:
            filtered_lightcurves[band] = np.concatenate(filtered_lightcurves[band],
                                                        axis=0)
            #parameters[band] = np.concatenate(parameters[band],
            #                                  axis=0)
        return filtered_lightcurves, parameters


class EmptyLightCurve(LightCurve):
    """Light curves of objects with no brightness"""
    def __init__(self, **kwargs):
        super(EmptyLightCurve, self).__init__(**kwargs)
        self.mag = 50

    def generate_lightcurves(self, n_lightcurves):
        params = {}
        lightcurves = {}
        for band in self.bands:
            params[band] = np.ones(shape=(n_lightcurves,))*self.mag
            params[band] = params[band].tolist()
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(self.observation_days[band])))*self.mag
        return lightcurves, params


from cepheids import *


if __name__ == "__main__":

    #observation_data = np.load("/home/tesla/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands-nf1-ne1-nr5-nn8855_CFHT-MegaCam_girz_days_limmag_100.pkl")
    #lc_path = "/home/tesla/rodrigo/supernovae_detection/francisco_sim/SNLS/all_bands_lightcurves.pkl"
    #param_path = "/home/tesla/rodrigo/supernovae_detection/francisco_sim/SNLS/all_bands_parameters.pkl"

    # To days and limmag per band
    observation_data = np.load(
        "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_days_limmag_sky100000.pkl")

    # supernovae light curves and parameters path (by Francisco's simulations)
    lc_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_LCs_500000.pkl"
    param_path = "/home/rodrigo/supernovae_detection/surveysim/pickles/MoriyaWindAcc_SNLS_bands_short-nf1-ne1-nr5-nn1500_CFHT-MegaCam_girz_params_500000.pkl"



    distr_path = "/home/rodrigo/supernovae_detection/object_distribution/"
    cepheids_path = "/home/toshiba/rodrigo/lsst_data/pablo_simulation/CEPH_SNLS_compress.pkl"
    M33_cepheids_path = "/home/toshiba/rodrigo/cepheids_lc/auto_selected_gp.pkl"
    bands = ["g", "r", "i"]
    observation_days = {}
    extrapolation_limits = {}
    band_ex = "g"
    limmag = {}
    for band in bands:
        extrapolation_limits[band] = [12, np.mean(observation_data[band]["limmag"])]
        observation_days[band] = observation_data[band]["days"]
        limmag[band] = observation_data[band]["limmag"]
    extend_fit = 0
    fit_limits = {"z": [17.5, 20+extend_fit],
                  "u": [16, 21+extend_fit],
                  "g": [19, 22+extend_fit],
                  "r": [18, 22+extend_fit],
                  "i": [18, 20+extend_fit]}

    light_curve = LightCurve(distribution_path=distr_path,
                             extrapolation_limit=extrapolation_limits,
                             fit_limits=fit_limits,
                             bands=bands,
                             observation_days=observation_days)

    # Supernovae test

    # supernovae = Supernovae(distribution_path=distr_path,
    #                         extrapolation_limit=extrapolation_limits,
    #                         fit_limits=fit_limits,
    #                         bands=bands,
    #                         observation_days=observation_days,
    #                         sn_lightcurves_path=lc_path,
    #                         sn_parameters_path=param_path,
    #                         limmag=limmag)
    #
    # sn_lightcurves, sn_parameters = supernovae.generate_lightcurves(n_lightcurves=5000)
    # print(" ----------------- Supernovae Lightcurves ------------------")
    # print("lc Keys:" + str(sn_lightcurves.keys()))
    # print("lc One key shape: " + str(sn_lightcurves[band_ex].shape))
    # print("lc One key examples: "+str(sn_lightcurves[band_ex][:5]))
    # print("params Keys:" + str(sn_parameters.keys()))
    # print("params One key len: " + str(len(sn_parameters[band_ex])))
    # print("params One key examples: "+str(sn_parameters[band_ex][:5]))
    # print("observation_days: "+str(observation_days["g"].shape))
    #
    # day_diff = []
    # for i in range(sn_lightcurves["g"].shape[0]):
    #     lc = sn_lightcurves["g"][i, :]
    #     #print(lc<30)
    #    # asdads
    #     days_over_30 = observation_days["g"][lc < 30]
    #     if len(days_over_30) != 0:
    #         day_dif = days_over_30[-1] - days_over_30[0]
    #         day_diff.append(day_dif)
    #     else:
    #         continue
    #
    # print(np.amax(day_diff))

    asteroids = Asteroid(distribution_path=distr_path,
                         extrapolation_limit=extrapolation_limits,
                         fit_limits=fit_limits,
                         bands=bands,
                         observation_days=observation_days,
                         sn_lightcurves_path=lc_path,
                         sn_parameters_path=param_path,
                         limmag=limmag)

    as_lightcurves, as_parameters = asteroids.generate_lightcurves(n_lightcurves=5000)
    print(" ----------------- Asteroids Lightcurves ------------------")
    print("lc Keys:" + str(as_lightcurves.keys()))
    print("lc One key shape: " + str(as_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(as_lightcurves[band_ex][0, ...]))
    print("asteroid magnitude: "+str(np.amin(as_lightcurves[band_ex][:5], axis=1)))
    print("params Keys:" + str(as_parameters.keys()))
    print("params One key len: " + str(len(as_parameters[band_ex])))
    print("params One key examples: "+str(as_parameters[band_ex][:5]))
    print("observation_days: "+str(observation_days["g"].shape))

    # Constant test
    asdasd
    constant = Constant(distribution_path=distr_path,
                        extrapolation_limit=extrapolation_limits,
                        fit_limits=fit_limits,
                        bands=bands,
                        observation_days=observation_days)

    const_lightcurves, const_parameters = constant.generate_lightcurves(5000)
    print(" ----------------- Constant Lightcurves ------------------")
    print("lc Keys:" + str(const_lightcurves.keys()))
    print("lc One key shape: " + str(const_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(const_lightcurves[band_ex][:5]))
    print("params Keys:" + str(const_parameters.keys()))
    print("params One key len: " + str(len(const_parameters[band_ex])))
    print("params One key examples: "+str(const_parameters[band_ex][:5]))

    # RRLyrae test

    rr_lyrae = RRLyrae(distribution_path=distr_path,
                       extrapolation_limit=extrapolation_limits,
                       fit_limits=fit_limits,
                       bands=bands,
                       observation_days=observation_days)

    rr_lightcurves, rr_parameters = rr_lyrae.generate_lightcurves(5000)
    print(" ----------------- RRLyrae Lightcurves ------------------")
    print("lc Keys:" + str(rr_lightcurves.keys()))
    print("lc One key shape: " + str(rr_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(rr_lightcurves[band_ex][:5]))
    print("params Keys:" + str(rr_parameters.keys()))
    print("params One key len: " + str(len(rr_parameters[band_ex])))
    print("params One key examples: "+str(rr_parameters[band_ex][:5]))

    # Empty lightcurve test

    empty = EmptyLightCurve(distribution_path=distr_path,
                            extrapolation_limit=extrapolation_limits,
                            fit_limits=fit_limits,
                            bands=bands,
                            observation_days=observation_days)

    emp_lightcurves, emp_parameters = empty.generate_lightcurves(5000)
    print(" ----------------- Empty Lightcurves ------------------")
    print("lc Keys:" + str(emp_lightcurves.keys()))
    print("lc One key shape: " + str(emp_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(emp_lightcurves[band_ex][:5]))
    print("params Keys:" + str(emp_parameters.keys()))
    print("params One key len: " + str(len(emp_parameters[band_ex])))
    print("params One key examples: "+str(emp_parameters[band_ex][:5]))

    # Cepheids lightcurve test
    ceph = M33Cepheids(distribution_path=distr_path,
                       extrapolation_limit=extrapolation_limits,
                       fit_limits=fit_limits,
                       bands=bands,
                       observation_days=observation_days,
                       M33_cepheids_path=M33_cepheids_path)

    ceph_lightcurves, ceph_parameters = ceph.generate_lightcurves(5000)
    print(" ----------------- M33 Cepheids Lightcurves ------------------")
    print("lc Keys:" + str(ceph_lightcurves.keys()))
    print("lc One key shape: " + str(ceph_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(ceph_lightcurves[band_ex][:5]))
    print("params Keys:" + str(ceph_parameters.keys()))
    print("params One key len: " + str(len(ceph_parameters[band_ex])))
    print("params One key examples: "+str(ceph_parameters[band_ex][:5]))


#    print(sn_lightcurves.shape)
#    print(const_lightcurves.shape)
#    print(rr_lightcurves.shape)

    if True:

        bins = np.linspace(12, 30, 300)
        band = "g"
        sn_n, bins = np.histogram(np.amin(sn_lightcurves[band], axis=1), bins=bins, normed=True)
        const_n, bins = np.histogram(const_lightcurves[band][:, 0], bins=bins, normed=True)
        rr_n, bins = np.histogram(np.reshape(rr_lightcurves[band], newshape=(-1,)), bins=bins, normed=True)
        limmag_n, bins = np.histogram(limmag[band], bins=bins, normed=True)
        ceph_n, _ = np.histogram(np.reshape(ceph_lightcurves[band], newshape=(-1,)), bins=bins, normed=True)

        plt.figure(figsize=(14, 7))
        plt.plot(bins[1:], sn_n, label="supernovae")
        plt.plot(bins[1:], rr_n, label="RR Lyrae")
        plt.plot(bins[1:], const_n, label="Constant")
        plt.plot(bins[1:], ceph_n, label="Cepheids")
        plt.plot(bins[1:], limmag_n, label="limmag")
        plt.xlabel("Magnitude", fontsize=16)
        plt.ylabel("Normalized Frequency", fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig("object_distr.png")
        plt.show()

    if False:
        mpl.style.use("default")
        plt.figure(figsize=(12, 7))
        bins = np.linspace(12, 30, 300)
        n_lc = []
        n_limmag = []
        for band in bands:
            n, _ = np.histogram(np.amin(sn_lightcurves[band], axis=1), bins=bins, density=True)
            n_lim, _ = np.histogram(limmag[band], bins=bins, density=True)
            n_lc.append(n)
            n_limmag.append(n_lim)
        for i, band in enumerate(bands):
            plt.plot(bins[1:], n_lc[i], "C"+str(i), label=band)
            plt.plot(bins[1:], n_limmag[i], "--C"+str(i), label="limmag "+band)
        plt.xlabel("Magnitude", fontsize=16)
        plt.title("Histogram of supernovae peaks per band")
        plt.ylabel("Normalized Frequency", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
    print("wena")
