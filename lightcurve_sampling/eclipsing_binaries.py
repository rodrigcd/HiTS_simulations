import numpy as np
from lightcurves import LightCurve
from os import listdir
from os.path import join
from tqdm import tqdm
import pickle
from scipy.interpolate import interp1d
from random import randint
import matplotlib.pyplot as plt

def Mag2Counts(lightcurves, airmass_per_obs=None, t_exp=225.0, zero_point=26.59, airmass_term=0.15):
    """light curve in magnitude to counts"""
    if not type(lightcurves) is np.ndarray:
        if airmass_per_obs is None:
            power = (lightcurves - zero_point) / -2.5
            return np.floor(np.power(10, power) * t_exp)
        else:
            power = (lightcurves - zero_point + airmass_term * (airmass_per_obs - 1)) / -2.5
            return np.floor(np.power(10, power) * t_exp)
    else:
        if airmass_per_obs is None:
            c_lightcurves = []
            for i in range(lightcurves.shape[0]):
                power = (lightcurves[i, :] - zero_point)/-2.5
                c_lightcurves.append(np.floor(np.multiply(np.power(10, power), t_exp))[np.newaxis, ...])
            c_lightcurves = np.concatenate(c_lightcurves, axis=0)
            return c_lightcurves
        else:
            c_lightcurves = []
            for i in range(lightcurves.shape[0]):
                power = (lightcurves[i, :] - zero_point + airmass_term * (airmass_per_obs - 1))/-2.5
                c_lightcurves.append(np.floor(np.power(10, power) * t_exp)[np.newaxis, ...])
            c_lightcurves = np.concatenate(c_lightcurves, axis=0)
            return c_lightcurves


def right_eb_criteria(eb_lightcurve, limmag, zero_point):
    indexes = np.argsort(eb_lightcurve)
    m1 = indexes[0]
    m0_0 = indexes[-1]
    m0_1 = indexes[-2]

    f1 = Mag2Counts(lightcurves=eb_lightcurve[m1],
                    zero_point=zero_point[m1],
                    t_exp=86.0)
    f0_0 = Mag2Counts(lightcurves=eb_lightcurve[m0_0],
                      zero_point=zero_point[m0_0],
                      t_exp=86.0)
    f0_1 = Mag2Counts(lightcurves=eb_lightcurve[m0_1],
                      zero_point=zero_point[m0_1],
                      t_exp=86.0)
    flim1 = Mag2Counts(lightcurves=limmag[m1],
                       zero_point=zero_point[m1],
                       t_exp=86.0)
    flim0_0 = Mag2Counts(lightcurves=limmag[m0_0],
                       zero_point=zero_point[m0_0],
                       t_exp=86.0)
    flim0_1 = Mag2Counts(lightcurves=limmag[m0_1],
                         zero_point=zero_point[m0_1],
                         t_exp=86.0)

    criteria_0 = (f1-f0_0)/np.sqrt(flim1**2+flim0_0**2) >1
    criteria_1 = (f1-f0_1)/np.sqrt(flim1**2+flim0_1**2) >1

    if criteria_0 and criteria_1:
        return True
    else:
        return False

def std_detection(eb_lightcurve):
    min_eclipses = [1, 5] # hardcoded
    mag = eb_lightcurve
    if np.sum(np.abs(mag-np.mean(mag))>np.std(mag))>int(np.random.uniform(low=min_eclipses[0],
                                                                          high=min_eclipses[1])):
        return True
    else:
        return False

def templates_to_pickle():
    path = "./lc_data/EB_templates"
    lc_templates = sorted(listdir(path))
    list_of_templates = []
    columns = ['phase', 'u', 'g', 'r', 'i', 'z', 'y']
    for template in tqdm(lc_templates):
        aux_dict = {}
        with open(join(path, template), 'r') as f:
            for line in f:
                if "Period" in line and "days" in line:
                    aux_dict["Period"] = np.float(line.split(" ")[-2])
                    break

        lc_data = np.loadtxt(join(path, template))
        for i, key in enumerate(columns):
            aux_dict[key] = lc_data[:, i]
        list_of_templates.append(aux_dict)
    with open('lc_data/eb_templates.pkl', 'wb') as f:
        pickle.dump(list_of_templates, f)


class EclipsingBinaries(LightCurve):

    def __init__(self, **kwargs):
        super(EclipsingBinaries, self).__init__(**kwargs)
        self.lc_path = kwargs["eb_path"]
        self.zero_point = kwargs["zero_point"]
        self.limmag = kwargs["limmag"]
        self.load_eb_templates()
        self.right_eb_count = 0

    def load_eb_templates(self):
        self.templates = np.load(self.lc_path)
        self.av_bands = list(self.templates[0].keys())[2:]
        self.interpolations = []
        for temp in self.templates:
            aux_dict = {}
            g_band_average = np.mean(temp["g"])
            for band in self.bands:
                x = np.append(temp["phase"], 1)
                y = np.append(temp[band], temp[band][0])
                y = 2*g_band_average - y
                aux_dict[band] = interp1d(x, y, kind='cubic')
            self.interpolations.append(aux_dict)

    def generate_single_lc(self, obs_days):
        while True:
            lc_id = randint(0, len(self.templates)-1)
            period = self.templates[lc_id]["Period"]
            interpolation = self.interpolations[lc_id]
            template_g_average = np.mean(self.templates[lc_id]["g"])

            mag_values = self.mag_generator.sample(1)
            mag = {}
            random_shift = np.random.random_sample()
            for band in self.bands:
                if len(obs_days[band]) == 0:
                    mag[band] = np.array([])
                    continue
                if not (band in self.av_bands):
                    raise ValueError('EB does not have '+band+' band')
                phase = np.mod(obs_days[band]+random_shift*period, period)*(1.0/period)
                lc = interpolation[band](phase)
                # mag[band] = lc/template_g_average*mag_values["g"]
                mag[band] = lc + (mag_values["g"]-1)
            # print(type(mag), type(self.limmag), type(self.zero_point))
            #if right_eb_criteria(mag["g"], self.limmag["g"], self.zero_point["g"]):
            if std_detection(mag["g"]):
                self.right_eb_count += 1
                new_mag = self.mag_generator.sample(1)
                for band in self.bands:
                    mag[band] = mag[band]-mag_values["g"]+new_mag["g"] 
                break

                plot_lc = False
                if plot_lc:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))
                    ax1.plot(self.templates[lc_id]["phase"], self.templates[lc_id][band], label="Real")
                    ax1.plot(self.templates[lc_id]["phase"],
                             interpolation[band](self.templates[lc_id]["phase"]), label="Invert_real")
                    ax1.plot(phase, lc, "o", label="invert_sampled")
                    ax2.plot(obs_days[band], lc, "o")
                    ax1.legend()
                    ax1.set_title(band+" period: "+str(period))
                    plt.show()
                    plt.close("all")
        return mag, period

    def generate_lightcurves(self, n_lightcurves,  obs_days=None, distr_limits=None, limmag=None, zero_point=None):
        if not obs_days:
            obs_days = self.observation_days
        elif len(obs_days)==0:
            return [array([]) for i in range(n_lightcurves)]
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
        self.limmag = limmag
        self.zero_point = zero_point

        lightcurves = {}
        params = {}
        lightcurves = {}
        lightcurves_list = []
        for i in range(n_lightcurves):
            # print(i)
            lightcurves_list.append(self.generate_single_lc(obs_days=obs_days))
        for band in self.bands:
            lightcurves[band] = []
            params[band] = []
            for i in range(n_lightcurves):
                lc, period = lightcurves_list[i]
                lightcurves[band].append(lc[band][np.newaxis, ...])
                params[band].append(np.array([np.mean(lc[band]), period]))
            lightcurves[band] = np.concatenate(lightcurves[band], axis=0)
            params[band] = params[band]
        return lightcurves, params


if __name__ == "__main__":
    # templates_to_pickle()
    cam_obs_cond = np.load("../real_obs/pickles/camera_and_obs_cond.pkl")
    print("n available fields: "+str(len(list(cam_obs_cond["obs_conditions"].keys()))))
    obs_cond = cam_obs_cond["obs_conditions"]["Field10"]
    print(obs_cond[0].keys())
    bands = ["g", "r", "i"]
    load_distribution = True
    extrapolation_limits = {}

    obs_days = {"g": [], "r": [], "i": []}
    limmag = {"g": [], "r": [], "i": []}
    zero_point = {"g": [], "r": [], "i": []}

    for epoch in obs_cond:
        obs_days[epoch["filter"]].append(epoch["obs_day"])
        limmag[epoch["filter"]].append(epoch["limmag5"])
        zero_point[epoch["filter"]].append(epoch["zero_point"])

    for band in bands:
        obs_days[band] = np.array(obs_days[band])
        limmag[band] = np.array(limmag[band])
        zero_point[band] = np.array(zero_point[band])
        ordered_index = np.argsort(obs_days[band])
        obs_days[band] = obs_days[band][ordered_index]
        limmag[band] = limmag[band][ordered_index]
        zero_point[band] = zero_point[band][ordered_index]
        extrapolation_limits[band] = [15, np.mean(limmag[band]) + np.std(limmag[band])*0.7]
        print("average limit of magnitude band "+band)
        print(np.mean(limmag[band]))

    eb_sampler = EclipsingBinaries(observation_days=obs_days,
                                   load_distr=load_distribution,
                                   extrapolation_limit=extrapolation_limits,
                                   bands=bands,
                                   eb_path="./lc_data/eb_templates.pkl",
                                   limmag=limmag,
                                   zero_point=zero_point)

    eb_sampler.generate_lightcurves(10000)
    print(eb_sampler.right_eb_count)
