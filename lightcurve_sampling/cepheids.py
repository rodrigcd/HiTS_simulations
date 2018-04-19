import numpy as np
from lightcurves import LightCurve
from random import randint
from sklearn.gaussian_process import GaussianProcessRegressor


class Cepheids(LightCurve):

    def __init__(self, **kwargs):
        super(Cepheids, self).__init__(**kwargs)
        self.lc_path = kwargs["cepheids_path"]

    def generate_lightcurves(self, n_lightcurves, re_sampled=True):
        data = np.load(self.lc_path, encoding="latin1")
        mag_samples = self.mag_generator.sample(n_lightcurves)
        lightcurves = {}
        params = {}
        for band in self.bands:
            lightcurves[band] = []
            params[band] = []
            for i in range(n_lightcurves):
                lc, period = data[band]["lightcurves"][i, :], data["periods"][i]
                if re_sampled:
                    lc += -np.mean(lc) + mag_samples[band][i]
                lightcurves[band].append(lc[np.newaxis, ...])
                params[band].append(np.array([mag_samples[band][i], period]))
            lightcurves[band] = np.concatenate(lightcurves[band], axis=0)
            #params[band] = np.concatenate(params[band], axis=0)
        return lightcurves, params


class M33Cepheids(LightCurve):

    def __init__(self, **kwargs):
        super(M33Cepheids, self).__init__(**kwargs)
        self.lc_path = kwargs["M33_cepheids_path"]
        self.lc_gps = np.load(self.lc_path, encoding="latin1")
        self.av_bands = list(self.lc_gps["stats"][0].keys())

    def generate_single_lc(self, obs_days):
        lc_id = randint(0, len(self.lc_gps["index"])-1)
        period = self.lc_gps["stats"][lc_id][self.av_bands[0]][2]

        gp = self.lc_gps["gps"][lc_id]
        #lightcurve = self.lc_gps["lightcurves"][lc_id]
        stats = self.lc_gps["stats"][lc_id]
        mag_values = self.mag_generator.sample(1)
        mag = {}
        for band in self.bands:
            if len(obs_days[band]) == 0:
                mag[band] = np.array([])
                #period = np.array([])
                continue
            if not (band in self.av_bands):
                raise ValueError('M33 survey does not have '+band+' band')
            phase = np.mod(obs_days[band]+np.random.random_sample()*period, period)*(1.0/period)
            lc = gp[band].predict(X=phase[:, np.newaxis])#, return_cov=True)#, n_samples=1, random_state=randint(0, 10000))
            mag[band] = lc*stats[band][1] + mag_values[band][0] #+ np.random.normal(loc=0, scale=np.sqrt(np.diag(cov)))*0.1
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
    from config_lightcurves import *
    import time
    cepheids = M33Cepheids(distribution_path=distr_path,
                           extrapolation_limit=extrapolation_limits,
                           fit_limits=fit_limits,
                           bands=bands,
                           observation_days=observation_days,
                           M33_cepheids_path=M33_cepheids_path)

    start = time.time()
    ceph_lightcurves, ceph_parameters = cepheids.generate_lightcurves(10)
    print(str(time.time() - start))
    print(" ----------------- Cepheids Lightcurves ------------------")
    print("lc Keys:" + str(ceph_lightcurves.keys()))
    print("lc One key shape: " + str(ceph_lightcurves[band_ex].shape))
    print("lc One key examples: "+str(ceph_lightcurves[band_ex][:5]))
    print("params Keys:" + str(ceph_parameters.keys()))
    print("params One key len: " + str(len(ceph_parameters[band_ex])))
    print("params One key examples: "+str(ceph_parameters[band_ex][:5]))




