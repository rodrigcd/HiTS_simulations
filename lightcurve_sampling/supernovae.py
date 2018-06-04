import numpy as np
from lightcurves import LightCurve

class TypeISupernovae(LightCurve):
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
        super(TypeISupernovae, self).__init__(**kwargs)
        self.lc_path = kwargs["type1_sn_lc_path"]
        self.limmag = kwargs["limmag"]
        self.param_path = kwargs["type1_sn_params_path"]
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

class TypeIISupernovae(LightCurve):
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
        super(TypeIISupernovae, self).__init__(**kwargs)
        self.lc_path = kwargs["type2_sn_lc_path"]
        self.limmag = kwargs["limmag"]
        self.param_path = kwargs["type2_sn_params_path"]
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
            # parameters[band] = np.concatenate(parameters[band],
            #                                  axis=0)
        return filtered_lightcurves, parameters


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
        self.type1_sn = TypeISupernovae(**kwargs)
        self.type2_sn = TypeIISupernovae(**kwargs)

    def generate_lightcurves(self, n_lightcurves, obs_days=None, distr_limits=None, field="Field01", limmag=None):
        s1, p1 = self.type1_sn.generate_lightcurves(n_lightcurves/2,
                                                    obs_days=obs_days,
                                                    distr_limits=distr_limits,
                                                    field=field,
                                                    limmag=limmag)
        s2, p2 = self.type2_sn.generate_lightcurves(n_lightcurves/2,
                                                    obs_days=obs_days,
                                                    distr_limits=distr_limits,
                                                    field=field,
                                                    limmag=limmag)

        random_index = np.arange(n_lightcurves)
        np.random.shuffle(random_index)

        p = {}
        s = {}
        for band in self.bands:
            s[band] = np.concatenate([s1[band], s2[band]])
            s[band] = s[band][random_index]
            p[band] = []
            join_p = p1[band] + p2[band]
            for index in random_index:
                p[band].append(join_p[index])

        return s, p