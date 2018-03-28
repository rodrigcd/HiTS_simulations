import numpy as np
import glob
import pickle


class ObsConditions(object):

    def __init__(self, **kwargs):
        self.sequence_length = kwargs["sequence_length"]
        self.data_path = kwargs["sn_data_path"]
        self.ccd_parameters_keys = kwargs["ccd_parameters_keys"]
        self.zero_points_path = kwargs["zero_point_path"]
        self.obs_conditions_keys = kwargs["obs_conditions_keys"]
        self.fixed_seed = 123
        self.sn_data = np.load(self.data_path)
        self.get_zero_points()
        self.match_ccds()
        self.generate_obs_conditions()
        self.save_data()

    def get_zero_points(self):
        zp_list = glob.glob(self.zero_points_path+"psm*")
        ccd_zp = []
        for i in range(62):
            ccd_zp.append({})
        for zp in zp_list:
            with open(zp, 'r') as f_aux:
                f_aux.readline()
                for i in range(62):
                    line = f_aux.readline().split(",")
                    zero_pont = line[6]
                    filter = line[5]
                    ccd_zp[i]["zp_"+filter] = np.float(zero_pont)*-1.0
        self.ccd_zp = ccd_zp

    def match_ccds(self):
        ccds = []
        sn_keys = list(self.sn_data.keys())
        headers = []
        for key in sn_keys:
            sn_headers = self.sn_data[key]["headers"]
            ccds.append(sn_headers["ccd_num"][0])
            headers.append(sn_headers)
        ccd_list, ccd_index = np.unique(ccds, return_index=True)
        ccd_params = []
        for index in ccd_index:
            aux_dict = {}
            for key in self.ccd_parameters_keys:
                aux_dict[key] = headers[index][key][0]
            aux_dict = {**aux_dict, **self.ccd_zp[aux_dict["ccd_num"]-1]}
            ccd_params.append(aux_dict)
            #print(aux_dict)
        self.ccd_params = ccd_params

    def generate_obs_conditions(self):
        obs_conditions = []
        sn_keys = list(self.sn_data.keys())
        for key in sn_keys:
            aux_dict = {}
            for obs_key in self.obs_conditions_keys:
                aux_dict[obs_key] = self.sn_data[key]["headers"][obs_key][:self.sequence_length]
            aux_dict["psf"] = self.sn_data[key]["psf"][..., :self.sequence_length]
            obs_conditions.append(aux_dict)
        self.obs_cond = obs_conditions

    def save_data(self):
        aux_dict = {"camera_params": self.ccd_params, "obs_conditions": self.obs_cond}
        with open('camera_and_obs_cond.pkl', 'wb') as f:
            pickle.dump(aux_dict, f)


if __name__ == "__main__":

    ccd_parameters_keys = ["ccd_num", "gain", "read_noise", "saturation"]
    obs_conditions_keys = ["sky_brightness", "airmass", "exp_time"]
    zero_points_path = "./hits_tables/"

    obs_conditions = ObsConditions(sequence_length=25,
                                   sn_data_path="sn_data.pkl",
                                   ccd_parameters_keys=ccd_parameters_keys,
                                   zero_point_path=zero_points_path,
                                   obs_conditions_keys=obs_conditions_keys)