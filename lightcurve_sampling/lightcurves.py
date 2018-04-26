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
#plt.switch_backend('agg')

class LightCurve(object):
    """Light curve main class"""
    def __init__(self, **kwargs):
        self.observation_days = kwargs["observation_days"]  # dictionary
        self.mag_generator = MagnitudeDistribution(**kwargs)
        self.bands = kwargs["bands"]

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None):
        """Returns a dictionary with keys per band with a list of light curves
        and dictionary with parameters of each light curve"""
        pass


class Asteroids(LightCurve):
    """Asteroid, params = magnitude"""
    def __init__(self, **kwargs):
        super(Asteroids, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None):
        if not obs_days:
            obs_days = self.observation_days
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
        lightcurves = {}
        mag_samples = self.mag_generator.sample(n_samples=n_lightcurves)
        time_samples = {}
        for band in self.bands:
            if len(obs_days[band]) == 0:
                mag_samples[band] = []
                lightcurves[band] = np.array([])
                continue
            time_samples[band] = np.random.randint(low=0, high=len(obs_days[band]))
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(obs_days[band])))*40.0
            lightcurves[band][:, time_samples[band]] = mag_samples[band]
            mag_samples[band] = mag_samples[band].tolist()
        return lightcurves, mag_samples


class NonVariable(LightCurve):
    """Constant object light curve, params = magnitude"""
    def __init__(self,  mag_limit=25, **kwargs):
        super(NonVariable, self).__init__(**kwargs)

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None):
        if not obs_days:
            obs_days = self.observation_days
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
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
            mag[band] = gen.generated(band, obs_days[band]+np.random.random_sample()*period, err=error)
        return mag, period

    def generate_lightcurves(self, n_lightcurves, re_sampled=True, obs_days=None, distr_limits=None):
        if not obs_days:
            obs_days = self.observation_days
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
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
            print("lightcurves type: "+str(type(lightcurves)))
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
        self.all_lightcurves = np.load(self.lc_path)
        self.all_params = np.load(self.param_path)

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None, field="Field01", limmag=None):
        if not obs_days:
            obs_days = self.observation_days
        if not limmag:
            limmag = self.limmag
        filtered_lightcurves = {}
        parameters = {}
        all_lightcurves_data = self.all_lightcurves[field]
        all_parameters_data = self.all_params[field]
        available_bands = list(all_lightcurves_data[0].keys())
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
            exp_range_per_band = []
            # for band in self.bands:
            #     if not (band in available_bands):
            #         continue
            #     mag_diff = limmag[band] - current_lightcurve[band]
            #     valid_index = np.where(mag_diff > 0)[0]
            #     detection_per_band.append(len(valid_index) > 0)
            #
            # G DETECTION
            mag_diff = limmag["g"] - current_lightcurve["g"]
            valid_index = np.where(mag_diff > 0)[0]
            detection_per_band.append(len(valid_index) > 0)

            # Adding exp time
            exp_diff = current_lightcurve["g"][0] - np.amin(current_lightcurve["g"])
            exp_range_per_band.append(exp_diff > 1)
            if any(detection_per_band) and any(exp_range_per_band):
                lc_count += 1
                for band in self.bands:
                    if not (band in available_bands):
                        filtered_lightcurves[band].append([])
                        parameters[band].append(np.array([]))
                        continue
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

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None):
        if not obs_days:
            obs_days = self.observation_days
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
        params = {}
        lightcurves = {}
        for band in self.bands:
            params[band] = np.ones(shape=(n_lightcurves,))*self.mag
            params[band] = params[band].tolist()
            lightcurves[band] = np.ones(shape=(n_lightcurves, len(obs_days[band])))*self.mag
        return lightcurves, params


from cepheids import *


if __name__ == "__main__":

    cam_obs_cond = np.load("../real_obs/pickles/camera_and_obs_cond.pkl")
    print("n available fields: "+str(len(list(cam_obs_cond["obs_conditions"].keys()))))
    obs_cond = cam_obs_cond["obs_conditions"]["Field01"]
    print(obs_cond[0].keys())
    bands = ["g", "r", "i"]
    load_distribution = True
    shift_limit = -1
    extrapolation_limits = {'g': [19, 25.602089154033994+shift_limit], 'r': [19, 25.029324900291915+shift_limit],
                            'i': [19, 24.45150161567846+shift_limit], 'z': [19, 23.122699702058064+shift_limit]}

    n_per_class = 500

    obs_days = {"g": [], "r": [], "i": []}
    limmag = {"g": [], "r": [], "i": []}

    for epoch in obs_cond:
        obs_days[epoch["filter"]].append(epoch["obs_day"])
        limmag[epoch["filter"]].append(epoch["limmag5"])

    for band in bands:
        obs_days[band] = np.array(obs_days[band])
        limmag[band] = np.array(limmag[band])
        ordered_index = np.argsort(obs_days[band])
        obs_days[band] = obs_days[band][ordered_index]
        limmag[band] = limmag[band][ordered_index]
        print("average limit of magnitude band "+band)
        print(np.mean(limmag[band]))


    #print(obs_days)
    #print(limmag)

    rrlyrae_gen = RRLyrae(observation_days=obs_days,
                          load_distr=load_distribution,
                          extrapolation_limit=extrapolation_limits,
                          bands=bands)
    rrlyrae_lc, rr_params = rrlyrae_gen.generate_lightcurves(n_per_class)

    asteroids_gen = Asteroids(observation_days=obs_days,
                                      load_distr=load_distribution,
                                      extrapolation_limit=extrapolation_limits,
                                      bands=bands)

    asteroids_lc, ast_params = asteroids_gen.generate_lightcurves(n_per_class, obs_days=obs_days, distr_limits=extrapolation_limits)

    constant_gen = NonVariable(observation_days=obs_days,
                          load_distr=load_distribution,
                          extrapolation_limit=extrapolation_limits,
                          bands=bands)


    # extrapolation_limits = {'g': [23, 23.1], 'r': [20, 21],
    #                         'i': [20, 21], 'z': [20, 21]}

    constant_lc, const_params = constant_gen.generate_lightcurves(n_per_class, obs_days=obs_days, distr_limits=extrapolation_limits)

    cepheids_gen = M33Cepheids(observation_days=obs_days,
                               load_distr=load_distribution,
                               extrapolation_limit=extrapolation_limits,
                               bands=bands,
                               M33_cepheids_path="./lc_data/cepheid_gps.pkl")

    cepheids_lc, ceph_params = cepheids_gen.generate_lightcurves(n_per_class)

    empty_gen = EmptyLightCurve(observation_days=obs_days,
                               load_distr=load_distribution,
                               extrapolation_limit=extrapolation_limits,
                               bands=bands)

    empty_lc, empty_params = empty_gen.generate_lightcurves(n_per_class)

    print(limmag.keys())
    
    # hits_lightcurves has 3000 light curves approx per field
    sn_gen = Supernovae(observation_days=obs_days,
                        load_distr=load_distribution,
                        extrapolation_limit=extrapolation_limits,
                        bands=bands,
                        limmag=limmag,
                        sn_lightcurves_path="/home/rodrigo/supernovae_detection/surveysim/pickles/hits_lightcurves.pkl",
                        sn_parameters_path="/home/rodrigo/supernovae_detection/surveysim/pickles/hits_params.pkl")
   
    sn_lc, sn_params = sn_gen.generate_lightcurves(n_per_class)
    print(sn_lc["g"].shape)

    plt.plot(obs_days["g"], rrlyrae_lc["g"][0, :], label="RRLyra")
    plt.plot(obs_days["g"], asteroids_lc["g"][0, :], label="Asteroid")
    plt.plot(obs_days["g"], constant_lc["g"][0, :], label="NonVariable")
    plt.plot(obs_days["g"], cepheids_lc["g"][0, :], label="Cepheid")
    plt.plot(obs_days["g"], sn_lc["g"][0, :], label="Supernovae")
    plt.plot(obs_days["g"], limmag["g"], label="limmag")


    plt.xlabel("obs day")
    plt.ylabel("magnitude")
    plt.ylim([27, 20])
    plt.legend()
    plt.savefig("images/examples.png")
    plt.show()

    for i in range(100):
        plt.plot(obs_days["g"], sn_lc["g"][i, :], label="RRLyra")
    plt.plot(obs_days["g"], limmag["g"], "o-k", label="limmag")
    plt.xlabel("obs day")
    plt.ylabel("magnitude")
    plt.ylim([27, 20])
    plt.savefig("images/supernovae_examples.png")
    plt.show()

    # Plot mag_distribution
    bins = np.linspace(16, 26, 100)
    rr_h, _ = np.histogram(np.mean(rrlyrae_lc["g"], axis=1), bins=bins, density=True)
    const_h, _ = np.histogram(np.mean(constant_lc["g"], axis=1), bins=bins, density=True)
    ceph_h, _ = np.histogram(np.mean(cepheids_lc["g"], axis=1), bins=bins, density=True)
    ast_h, _ = np.histogram(np.amin(asteroids_lc["g"], axis=1), bins=bins, density=True)
    sn_h, _ = np.histogram(np.min(sn_lc["g"], axis=1), bins=bins, density=True)
    lim_h, _ = np.histogram(limmag["g"], bins=bins, density=True)

    plt.plot(bins[1:], rr_h, label="RRLyra")
    plt.plot(bins[1:], const_h, label="Const")
    plt.plot(bins[1:], ceph_h, label="Ceph")
    plt.plot(bins[1:], ast_h, label="Ast")
    plt.plot(bins[1:], sn_h, label="SN")
    plt.plot(bins[1:], lim_h, label="limm")
    plt.legend()
    plt.savefig("images/sampling_distr.png")
    #plt.xlim([30, 15])
    plt.show()

    field_list = list(cam_obs_cond["obs_conditions"].keys())
    for field in field_list:
        cond = cam_obs_cond["obs_conditions"][field]
        obs_days = []
        limmag = []
        for epoch in cond:
            if epoch["filter"] == "g":
                obs_days.append(epoch["obs_day"])
                limmag.append(epoch["limmag5"])
        print(field+" average limmag: "+str(np.mean(limmag)) + ", std: "+str(np.std(limmag)))


    print("wena")
