import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors.kde import KernelDensity
from gatspy.datasets import fetch_rrlyrae, RRLyraeGenerated
import warnings
import pandas
from sklearn.neighbors.kde import KernelDensity
from custom_distribution import MagnitudeDistribution
import matplotlib as mpl
warnings.filterwarnings("ignore")


class LightCurve(object):
    """Light curve main class"""
    def __init__(self, **kwargs):
        self.observation_days = kwargs["observation_days"]  # dictionary
        self.mag_generator = MagnitudeDistribution(**kwargs)
        self.bands = kwargs["bands"]

    def generate_lightcurves(self, n_lightcurves, obs_days=None):
        """Returns a dictionary with keys per band with a list of light curves
        and dictionary with parameters of each light curve"""
        pass


class Asteroids(LightCurve):
    """Asteroid, params = magnitude"""
    def __init__(self, **kwargs):
        super(Asteroids, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves, obs_days=None):
        if not obs_days:
            obs_days = self.observation_days
        lightcurves = {}
        mag_samples = self.mag_generator.sample(n_samples=n_lightcurves)
        time_samples = {}
        for band in self.bands:
            time_samples[band] = np.random.randint(low=0, high=len(obs_days[band]))
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(obs_days[band])))*40.0
            lightcurves[band][:, time_samples[band]] = mag_samples[band]
            mag_samples[band] = mag_samples[band].tolist()
        return lightcurves, mag_samples


class Constant(LightCurve):
    """Constant object light curve, params = magnitude"""
    def __init__(self,  mag_limit=25, **kwargs):
        super(Constant, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves, obs_days=None):
        if not obs_days:
            obs_days = self.observation_days
        lightcurves = {}
        mag_samples = self.mag_generator.sample(n_samples=n_lightcurves)
        for band in self.bands:
            lightcurves[band] = np.repeat(mag_samples[band][..., np.newaxis],
                                          len(obs_days[band]),
                                          axis=1)
            # np.random.normal(loc=0, scale=0.01, size=lightcurves.shape) No noise
            mag_samples[band] = mag_samples[band].tolist()
        return lightcurves, mag_samples


class RRLyrae(LightCurve):
    """RRLyrae object light curve, params = mean(magnitude) sampled from distribution"""
    def __init__(self, **kwargs):
        super(RRLyrae, self).__init__(**kwargs)
        self.id_count = 0
        self.rrlyrae = fetch_rrlyrae()
        self.std_values = pandas.read_csv("lc_data/RRLYR_HiTS_std.csv")["Std"].values
        silverman_sigma = np.std(self.std_values)*np.power((4.0/3.0/len(self.std_values)), (1.0/5.0))
        print(silverman_sigma)
        self.kde_sampler = KernelDensity(kernel="gaussian", bandwidth=silverman_sigma).fit(self.std_values[..., np.newaxis])

    def generate_single_lc(self, obs_days):
        lcid = self.rrlyrae.ids[self.id_count]
        period = self.rrlyrae.get_metadata(lcid)["P"]
        self.id_count += 1
        random_state_int = np.random.randint(low=0, high=10000, size=1)
        if self.id_count >= len(self.rrlyrae.ids):
            self.id_count = 0
        gen = RRLyraeGenerated(lcid, random_state=random_state_int)
        mag = {}
        error = np.clip(self.kde_sampler.sample(n_samples=1)[0, 0], a_min=0.011, a_max=None)
        for band in self.bands:
            mag[band] = gen.generated(band, obs_days[band], err=error)
        return mag, period

    def generate_lightcurves(self, n_lightcurves, re_sampled=True, obs_days=None):
        if not obs_days:
            obs_days = self.observation_days
        lightcurves = {}
        params = {}
        lightcurves_list = []
        mag_samples = self.mag_generator.sample(n_lightcurves)
        for i in range(n_lightcurves):
            lightcurves_list.append(self.generate_single_lc(obs_days=obs_days))
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

    def generate_lightcurves(self, n_lightcurves, obs_days=None):
        if not obs_days:
            obs_days = self.observation_days
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

    def generate_lightcurves(self, n_lightcurves, obs_days=None):
        if not obs_days:
            obs_days = self.observation_days
        params = {}
        lightcurves = {}
        for band in self.bands:
            params[band] = np.ones(shape=(n_lightcurves,))*self.mag
            params[band] = params[band].tolist()
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(obs_days[band])))*self.mag
        return lightcurves, params


from cepheids import *


if __name__ == "__main__":

    cam_obs_cond = np.load("../real_obs/camera_and_obs_cond.pkl")
    obs_cond = cam_obs_cond["obs_conditions"]["Field01"]
    bands = ["g", "r", "i"]
    load_distribution = True
    extrapolation_limits = {'g': [12, 25.602089154033994], 'r': [12, 25.029324900291915], 'i': [12, 24.45150161567846], 'z': [12, 23.122699702058064]}

    obs_days = {"g": [], "r": [], "i": []}

    for epoch in obs_cond:
        obs_days[epoch["filter"]] = epoch["obs_day"]

    rrlyrae_gen = RRLyrae(observation_days=obs_days,
                          load_distr=load_distribution,
                          extrapolation_limit=extrapolation_limits,
                          bands=bands)

    rrlyrae_gen.generate_lightcurves(100)



    print("wena")