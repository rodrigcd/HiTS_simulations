import numpy as np
from lightcurves import LightCurve
from os import listdir
from os.path import join
from tqdm import tqdm
import pickle
from scipy.interpolate import interp1d
from random import randint
import matplotlib.pyplot as plt

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
        self.load_eb_templates()

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
        lc_id = randint(0, len(self.templates)-1)
        period = self.templates[lc_id]["Period"]
        interpolation = self.interpolations[lc_id]
        template_g_average = np.mean(self.templates[lc_id]["g"])

        mag_values = self.mag_generator.sample(1)
        mag = {}
        for band in self.bands:
            if len(obs_days[band]) == 0:
                mag[band] = np.array([])
                continue
            if not (band in self.av_bands):
                raise ValueError('M33 survey does not have '+band+' band')
            phase = np.mod(obs_days[band]+np.random.random_sample()*period, period)*(1.0/period)
            lc = interpolation[band](phase)
            # mag[band] = lc/template_g_average*mag_values["g"]
            mag[band] = lc + (mag_values["g"]-1)
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,7))
            # ax1.plot(self.templates[lc_id]["phase"], self.templates[lc_id][band], label="Real")
            # ax1.plot(self.templates[lc_id]["phase"],
            #          interpolation[band](self.templates[lc_id]["phase"]), label="Invert_real")
            # ax1.plot(phase, lc, "o", label="invert_sampled")
            # ax2.plot(obs_days[band], lc, "o")
            # ax1.legend()
            # ax1.set_title(band)
            # plt.show()
            # plt.close("all")
        return mag, period

    def generate_lightcurves(self, n_lightcurves,  obs_days=None, distr_limits=None):
        if not obs_days:
            obs_days = self.observation_days
        if distr_limits:
            self.mag_generator.set_extrapolation_limits(distr_limits)
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

    eb_sampler = EclipsingBinaries(observation_days=obs_days,
                                   load_distr=load_distribution,
                                   extrapolation_limit=extrapolation_limits,
                                   bands=bands,
                                   eb_path="./lc_data/eb_templates.pkl")

    eb_sampler.generate_lightcurves(100)
