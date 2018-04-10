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
        self.observation_days = kwargs["observation_days"]
        self.limmag = kwargs["limmag"]

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

    def generate_lightcurves(self, n_lightcurves_per_class, shuffled=True):
        unique_labels, labels_counts = np.unique(self.requested_lightcurves_labels,
                                                 return_counts=True)
        lightcurves_list = []
        parameters_list = []
        labels = []
        lc_type = []
        n_lc_per_type = {}
        for i, lightcurve_obj in enumerate(self.lightcurve_objects):
            print("Creating "+lightcurve_obj.__class__.__name__+" light curves")
            n_same_label = labels_counts[unique_labels == self.requested_lightcurves_labels[i]]
            n_lc = int(np.round(np.true_divide(n_lightcurves_per_class, n_same_label)))
            lc, params = lightcurve_obj.generate_lightcurves(n_lc)
            lightcurves_list.append(lc)
            parameters_list.append(params)
            # print(type(params["g"]))
            labels.append(self.requested_lightcurves_labels[i] * np.ones(shape=(n_lc,)))
            lc_type += [lightcurve_obj.__class__.__name__] * n_lc
            print(str(n_lc) + " " + lightcurve_obj.__class__.__name__ + " light curves")
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

        print(lightcurves["g"].shape)

        for band in self.bands:
            lightcurves[band] = lightcurves[band][shuffled_index, ...]
            parameters[band] = [parameters[band][i] for i in shuffled_index]
        labels = labels[shuffled_index]
        lc_type = lc_type[shuffled_index]
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

        hdf5_file = h5py.File(self.save_path + self.file_name + ".hdf5", "w")
        dt = h5py.special_dtype(vlen=unicode)
        lc_group = hdf5_file.create_group("lightcurves")
        obs_days_group = hdf5_file.create_group("obs_days")
        limmag_group = hdf5_file.create_group("limmag")
        n_per_type_group = hdf5_file.create_group("n_lc_per_type")
        for band in self.bands:
            lc_group.create_dataset(band, data=lightcurves[band])
            obs_days_group.create_dataset(band, data=self.observation_days[band])
            limmag_group.create_dataset(band, data=self.limmag[band])
        for object_name in self.requested_lightcurves:
            n_per_type_group.create_dataset(object_name, data=n_lc_per_type[object_name])
        hdf5_file.create_dataset("labels", data=labels)
        hdf5_file.create_dataset("lc_type", data=lc_type, dtype=dt)
        hdf5_file.create_dataset("lc_id", data=lc_id)
        # hdf5_file.create_dataset("n_lc_per_type", data=n_lc_per_type[self.bands[0]])
        # hdf5_file.create_dataset("obs_days", data=self.observation_days)
        # hdf5_file.create_dataset("limmag", data=self.limmag)

        pickle.dump({"parameters": parameters},
                    open(self.save_path + self.file_name + ".pkl", "wb"),
                    protocol=2)

        return lightcurves, labels, lc_type, parameters


if __name__ == "__main__":

    if sys.argv[1] == "SNLS":
        from config_lightcurves import *
    elif sys.argv[1] == "KMTNet":
        from config_lightcurves_KMTNet import *

    lc_database = LightCurveDatabase(requested_lightcurves=requested_lightcurve,
                                     requested_lightcurves_labels=requested_lightcurve_labels,
                                     observation_days=observation_days,
                                     sn_lightcurves_path=lc_path,
                                     sn_parameters_path=param_path,
                                     limmag=limmag,
                                     fit_limits=fit_limits,
                                     distribution_path=distr_path,
                                     extrapolation_limit=extrapolation_limits,
                                     bands=bands,
                                     save_path=save_path,
                                     file_name=file_name,
                                     cepheids_path=cepheids_path,
                                     M33_cepheids_path=M33_cepheids_path)

    lc, labels, lc_type, params = lc_database.generate_lightcurves(n_lightcurves_per_class=n_per_class)
    print("-----Lightcurves shapes-----")
    print("lightcurves keys: "+str(lc.keys()))
    print("lightcurves one key shape: "+str(lc["g"].shape))
    print("lightcurves one key shape: "+str(lc["r"].shape))
    print("lightcurves one key shape: "+str(lc["i"].shape))
    #print("lightcurves one key example: "+str(lc["g"][:3, :]))
    print("-----Labels shapes-----")
    print("labels shape: "+str(labels.shape))
    #print("labels example: "+str(labels))
    print("-----Types shape-----")
    print("lc_type shape: "+str(lc_type.shape))
    #print("lc_type example: "+str(lc_type))
    print("-----Parameters shapes-----")
    print("parameters keys: "+str(params.keys()))
    print("parameters one key len: "+str(len(params["g"])))