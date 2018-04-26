import numpy as np
from lightcurves import LightCurve
import pickle
import sys
import h5py


class LightCurveDatabase(object):

    def __init__(self, **kwargs):
        self.available_lightcurves = [cls for cls in globals()['LightCurve'].__subclasses__()]
        #print(type(self.available_lightcurves[0]))
        self.available_lightcurves_names = [cls.__name__ for cls in self.available_lightcurves]
        self.requested_lightcurves = kwargs["requested_lightcurves"]
        self.requested_lightcurves_labels = kwargs["requested_lightcurves_labels"]
        self.bands = kwargs["bands"]
        self.save_path = kwargs["save_path"]
        self.file_name = kwargs["file_name"]
        self.n_std_limmag = kwargs["n_std_limmag"]
        self.camera_and_obs_cond_path = kwargs["camera_and_obs_cond_path"]
        self.camera_and_obs_cond = np.load(self.camera_and_obs_cond_path)

        # ------ Just to initialize light curves objects, hardcoded :( ------
        field01_days = []
        field01_limmag = []
        self.field_list = list(self.camera_and_obs_cond["obs_conditions"].keys())
        field01 = self.camera_and_obs_cond["obs_conditions"]["Field01"]
        for epoch in field01:
            field01_days.append(epoch["obs_day"])
            field01_limmag.append(epoch["limmag5"])

        ordered_index = np.argsort(field01_days)
        field01_days = np.array(field01_days)[ordered_index]
        field01_limmag = np.array(field01_limmag)[ordered_index]
        # Standard extrapolation limits
        shift_limit = -1
        extrapolation_limits = {'g': [19, 25.602089154033994 + shift_limit],
                                'r': [19, 25.029324900291915 + shift_limit],
                                'i': [19, 24.45150161567846 + shift_limit],
                                'z': [19, 23.122699702058064 + shift_limit]}
        kwargs["observation_days"] = field01_days
        kwargs["load_distr"] = True
        kwargs["extrapolation_limit"] = extrapolation_limits
        kwargs["limmag"] = field01_limmag

        print("----- Available Light Curves -----")
        for cls in self.available_lightcurves:
            print(cls.__name__)
        print("----- Requested Light Curves and Labels -----")
        print(self.requested_lightcurves)
        print(self.requested_lightcurves_labels)

        self.lightcurve_objects = []
        for requested_name in self.requested_lightcurves:
            if requested_name in self.available_lightcurves_names:
                index = self.available_lightcurves_names.index(requested_name)
                self.lightcurve_objects.append(self.available_lightcurves[index](**kwargs))
            else:
                print(requested_name + " not available")
                continue

    def generate_lightcurves(self, n_lightcurves_per_class_per_field, shuffled=True):
        unique_labels, labels_counts = np.unique(self.requested_lightcurves_labels,
                                                 return_counts=True)

        hdf5_file = h5py.File(self.save_path + self.file_name + ".hdf5", "w")
        field_parameters = {}
        for i_field, field in enumerate(self.field_list):
            print("Simulating for field "+field)
            # if field == "Field04":
            #   continue

            field_group = hdf5_file.create_group(field)

            obs_cond = self.camera_and_obs_cond["obs_conditions"][field]
            obs_days = {"g": [], "r": [], "i": []}
            limmag = {"g": [], "r": [], "i": []}
            extrapolation_limits = {}

            for epoch in obs_cond:
                obs_days[epoch["filter"]].append(epoch["obs_day"])
                limmag[epoch["filter"]].append(epoch["limmag5"])

            for band in self.bands:
                obs_days[band] = np.array(obs_days[band])
                limmag[band] = np.array(limmag[band])
                ordered_index = np.argsort(obs_days[band])
                obs_days[band] = obs_days[band][ordered_index]
                limmag[band] = limmag[band][ordered_index]
                extrapolation_limits[band] = [19, np.mean(limmag[band]) + np.std(limmag[band])*self.n_std_limmag]

            lightcurves_list = []
            parameters_list = []
            labels = []
            lc_type = []
            n_lc_per_type = {}

            for i, lightcurve_obj in enumerate(self.lightcurve_objects):
                class_name = lightcurve_obj.__class__.__name__

                n_same_label = labels_counts[unique_labels == self.requested_lightcurves_labels[i]]
                n_lc = int(np.round(np.true_divide(n_lightcurves_per_class_per_field, n_same_label)))
                if class_name == "Supernovae":
                    lc, params = lightcurve_obj.generate_lightcurves(n_lightcurves=n_lc,
                                                                     obs_days=obs_days,
                                                                     distr_limits=extrapolation_limits,
                                                                     field=field,
                                                                     limmag=limmag)
                else:
                    lc, params = lightcurve_obj.generate_lightcurves(n_lightcurves=n_lc,
                                                                     obs_days=obs_days,
                                                                     distr_limits=extrapolation_limits)
                lightcurves_list.append(lc)
                parameters_list.append(params)
                # print(type(params["g"]))
                labels.append(self.requested_lightcurves_labels[i] * np.ones(shape=(n_lc,)))
                lc_type += [lightcurve_obj.__class__.__name__] * n_lc
                #print(str(n_lc) + " " + lightcurve_obj.__class__.__name__ + " light curves")
                if i_field == 0:
                    print("Sampling " + lightcurve_obj.__class__.__name__ + " light curves")
                    print(str(len(lc["g"])) + " " + lightcurve_obj.__class__.__name__ + " light curves generated")
                n_lc_per_type[lightcurve_obj.__class__.__name__] = n_lc

            lightcurves = {}
            parameters = {}
            # Merge dictionaries
            for band in self.bands:
                lightcurves[band] = []
                parameters[band] = []
                for i, lc in enumerate(lightcurves_list):
                    lightcurves[band].append(lc[band])
                    parameters[band] += parameters_list[i][band]
                    # print(lightcurves[band][-1].shape)
                lightcurves[band] = np.concatenate(lightcurves[band], axis=0)
                #parameters[band] = np.array(parameters[band])
            # print(parameters["g"])

            # lightcurves = np.concatenate(lightcurves, axis=0)
            labels = np.concatenate(labels, axis=0)
            lc_type = np.array(lc_type)
            # parameters = np.array(parameters)

            if not shuffled:
                return lightcurves, labels, lc_type, parameters

            # Shuffling labels
            label_index = []
            for label in unique_labels:
                label_index.append(np.where(labels == label)[0])
            shuffled_index = np.concatenate(label_index, axis=0)
            np.random.shuffle(shuffled_index)

            print("lightcurves shape: "+str(lightcurves["g"].shape))

            for band in self.bands:
                lightcurves[band] = lightcurves[band][shuffled_index, ...]
                parameters[band] = [parameters[band][i] for i in shuffled_index]
            labels = labels[shuffled_index]
            lc_type = lc_type[shuffled_index]
            lc_type = np.array(lc_type).astype('S')
            lc_id = np.arange(len(labels))

            print("saving light curves")
            # pickle.dump({"lightcurves": lightcurves,
            #              "labels": labels,
            #              "lc_type": lc_type,
            #              "parameters": parameters,
            #              "lc_id": lc_id,
            #              "n_lc_per_type": n_lc_per_type,
            #              "obs_days": self.observation_days,
            #              "limmag": self.limmag},
            #             open(self.save_path + self.file_name + ".pkl", "wb"), protocol=2)

            dt = h5py.special_dtype(vlen=str)
            lc_group = field_group.create_group("lightcurves")
            obs_days_group = field_group.create_group("obs_days")
            limmag_group = field_group.create_group("limmag")
            n_per_type_group = field_group.create_group("n_lc_per_type")
            for band in self.bands:
                lc_group.create_dataset(band, data=lightcurves[band])
                obs_days_group.create_dataset(band, data=obs_days[band])
                limmag_group.create_dataset(band, data=limmag[band])
            for object_name in self.requested_lightcurves:
                n_per_type_group.create_dataset(object_name, data=n_lc_per_type[object_name])
            field_group.create_dataset("labels", data=labels)
            field_group.create_dataset("lc_type", data=lc_type, dtype=dt)
            field_group.create_dataset("lc_id", data=lc_id)
            # hdf5_file.create_dataset("n_lc_per_type", data=n_lc_per_type[self.bands[0]])
            # hdf5_file.create_dataset("obs_days", data=self.observation_days)
            # hdf5_file.create_dataset("limmag", data=self.limmag)
            field_parameters[field] = parameters

        print("Total number of lightcurves per class: "+str(len(self.field_list)*n_lightcurves_per_class_per_field))
        print("Total number of lightcurves: "+str(len(self.field_list)*n_lightcurves_per_class_per_field*len(self.requested_lightcurves)))

        pickle.dump(field_parameters,
                    open(self.save_path + self.file_name + ".pkl", "wb"),
                    protocol=2)

        # return lightcurves, labels, lc_type, parameters


if __name__ == "__main__":

    from config_lightcurves import *

    lc_database = LightCurveDatabase(requested_lightcurves=requested_lightcurve,
                                     requested_lightcurves_labels=requested_lightcurve_labels,
                                     file_name=file_name,
                                     save_path=save_path,
                                     bands=bands,
                                     camera_and_obs_cond_path=camera_and_obs_cond_path,
                                     sn_lightcurves_path=sn_lightcurves_path,
                                     sn_parameters_path=sn_parameters_path,
                                     M33_cepheids_path=M33_cepheids_path,
                                     n_std_limmag=n_std_limmag,
                                     eb_path=eb_path)

    lc_database.generate_lightcurves(n_lightcurves_per_class_per_field=n_per_class_per_field, shuffled=True)
