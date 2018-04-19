import numpy as np


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
                c_lightcurves.append(np.floor(np.power(10, power) * t_exp)[np.newaxis, ...])
            c_lightcurves = np.concatenate(c_lightcurves, axis=0)
            return c_lightcurves
        else:
            c_lightcurves = []
            for i in range(lightcurves.shape[0]):
                power = (lightcurves[i, :] - zero_point + airmass_term * (airmass_per_obs - 1))/-2.5
                c_lightcurves.append(np.floor(np.power(10, power) * t_exp)[np.newaxis, ...])
            c_lightcurves = np.concatenate(c_lightcurves, axis=0)
            return c_lightcurves


def Count2Mag(counts, airmass_per_obs=None, t_exp=225.0, zero_point=26.59, airmass_term=0.15):
    if not type(counts) is np.ndarray:
        if airmass_per_obs is None:
            m = zero_point - 2.5*np.log10(counts/t_exp)
            return m
        else:
            m = zero_point - 2.5*np.log10(counts/t_exp) - airmass_term * (airmass_per_obs - 1)
            return m

    else:
        if airmass_per_obs is None:
            m_lightcurves = []
            for i in range(counts.shape[0]):
                m = zero_point - 2.5 * np.log10((counts[i, :] / t_exp) + 10e-4)
                m_lightcurves.append(m[np.newaxis, ...])

            m_lightcurves = np.concatenate(m_lightcurves, axis=0)
            return m_lightcurves
        else:
            m_lightcurves = []
            for i in range(counts.shape[0]):
                m = zero_point - 2.5 * np.log10((counts[i, :] / t_exp) + 10e-4) - airmass_term * (airmass_per_obs - 1)
                m_lightcurves.append(m[np.newaxis, ...])

            m_lightcurves = np.concatenate(m_lightcurves, axis=0)
            return m_lightcurves
