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
        self.surveysim_data_path = kwargs["surveysim_data_path"]
        self.per_field_epoch = kwargs["per_field_epoch"]
        self.field_cond_path = kwargs["field_cond_path"]
        self.npy_keys = kwargs["npy_keys"]
        self.fixed_seed = 123
        self.sn_data = np.load(self.data_path)
        self.get_zero_points()
        self.match_ccds()
        if not per_field_epoch:
            self.generate_obs_conditions()
            self.save_data()
        else:
            self.read_fields()
            self.save_data()
            self.surveysim_data()

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
        ccd_params = {}
        for index in ccd_index:
            aux_dict = {}
            for key in self.ccd_parameters_keys:
                aux_dict[key] = headers[index][key][0]
            aux_dict = {**aux_dict, **self.ccd_zp[aux_dict["ccd_num"]-1]}
            ccd_params["CCD"+str(aux_dict["ccd_num"])] = aux_dict
            #print(aux_dict)
        self.ccd_params = ccd_params

    def generate_obs_conditions(self):
        obs_conditions = []
        sn_keys = list(self.sn_data.keys())
        self.unique_obs_days = []
        for key in sn_keys:
            aux_dict = {}
            for obs_key in self.obs_conditions_keys:
                aux_dict[obs_key] = self.sn_data[key]["headers"][obs_key][:self.sequence_length]
            aux_dict["psf"] = self.sn_data[key]["psf"][..., :self.sequence_length]
            self.unique_obs_days.append(self.sn_data[key]["headers"]["obs_days"][:self.sequence_length])
            obs_conditions.append(aux_dict)
        self.obs_cond = obs_conditions
        self.unique_obs_days = np.unique(np.concatenate(self.unique_obs_days, axis=0))
        #print(len(self.unique_obs_days), self.unique_obs_days.shape)
        #print(self.unique_obs_days)

    def read_fields(self):
        info_list = sorted(glob.glob(self.field_cond_path+"*.npy"), key=str.lower)
        self.obs_cond = {}
        for info in info_list:
            field, epoch = np.array(info.split("_")[2:4]).astype(np.int)
            info_array = np.load(info)
            aux_dict = {}
            #print(field, epoch)
            for i, key in enumerate(self.npy_keys):
                try:
                    if key == "FILTER":
                        aux_dict[self.obs_conditions_keys[i]] = str(info_array[key])[2]
                    else:
                        aux_dict[self.obs_conditions_keys[i]] = np.float(info_array[key])
                except:
                    aux_dict[self.obs_conditions_keys[i]] = \
                        self.obs_cond["Field"+str(field).zfill(2)][-1][self.obs_conditions_keys[i]]
                    print("field "+str(field)+", epoch "+str(epoch))
                    print("does not have key "+key+", replacing with previous value "+str(aux_dict[self.obs_conditions_keys[i]]))

            if epoch == 1:
                self.obs_cond["Field"+str(field).zfill(2)] = [aux_dict, ]
            else:
                self.obs_cond["Field"+str(field).zfill(2)].append(aux_dict)

    def save_data(self):
        aux_dict = {"camera_params": self.ccd_params, "obs_conditions": self.obs_cond}
        with open('camera_and_obs_cond.pkl', 'wb') as f:
            pickle.dump(aux_dict, f)

    def surveysim_data(self):

        #aux_file = open(self.surveysim_data_path+"SN_HiTS.dat", "w")
        #aux_file.write("MJD FILTER EXPTIME AIRMASS\n")
        #for day in self.unique_obs_days:
        #    aux_file.write(str(day)+" "+band+"\n")
        #aux_file.close()
        for field, epoch_list in self.obs_cond.items():
            #print(field)
            aux_file = open(self.surveysim_data_path+"HiTS_fields/"+field+".dat", "w")
            aux_file.write("MJD FILTER EXPTIME AIRMASS EPOCH\n")
            n_g = 0
            for epoch in epoch_list:
                day = epoch["obs_day"]
                exp_time = epoch["exp_time"]
                airmass = epoch["airmass"]
                aux_epoch = epoch["epoch"]
                if epoch["filter"] == "g":
                    n_g += 1
                line = str(day)+" "+epoch["filter"]+" "+str(exp_time)+" "+str(airmass)+" "+str(aux_epoch)
                aux_file.write(line+"\n")
            print(n_g)
            aux_file.close()


if __name__ == "__main__":

    ccd_parameters_keys = ["ccd_num", "gain", "read_noise", "saturation"]
    obs_conditions_keys = ["sky_brightness", "airmass", "exp_time", "obs_day",
                           "filter", "seeing", "epoch", "limmag5", "limmag3", "zero_point"]
    npy_keys = ["BACK_LEVEL", "AIRMASS", "EXP_TIME", "MJD", "FILTER", "SEEING", "EPOCH",
                "LIMIT_MAG_EA5", "LIMIT_MAG_EA3", "ZP_PS"]
    zero_points_path = "./HiTS_tables/"
    fields_cond_path = "./Blind15A_info/"
    surveysim_data_path = "/home/rodrigo/supernovae_detection/surveysim/obsplans/"
    per_field_epoch = True

    obs_conditions = ObsConditions(sequence_length=25,
                                   sn_data_path="sn_data.pkl",
                                   ccd_parameters_keys=ccd_parameters_keys,
                                   zero_point_path=zero_points_path,
                                   obs_conditions_keys=obs_conditions_keys,
                                   npy_keys=npy_keys,
                                   surveysim_data_path=surveysim_data_path,
                                   per_field_epoch=per_field_epoch,
                                   field_cond_path=fields_cond_path)

    # obs_conditions.surveysim_data()