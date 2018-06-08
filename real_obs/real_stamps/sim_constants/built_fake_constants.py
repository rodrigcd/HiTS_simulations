import numpy as np
import h5py
import pickle

image_path = "/home/rcarrasco/simulated_data/image_sequences/complete_may30_erf_distr2500.hdf5"
data = h5py.File(image_path, "r")
n_samples = 30

for field in list(data.keys()):
    print(field)
    lc_types = data[field]["lc_type"][:]
    for i, lc_type in enumerate(lc_types):
        if lc_type == "NonVariable":
            aux_dict = {}
            image = data[field]["images"]["g"][i, ...]
            days = data[field]["obs_cond"]["obs_day"]["g"][:]
            print(image.shape)
            print(days.shape)
            aux_dict["stamp"] = np.rollaxis(image, axis=2)
            aux_dict["time"] = {"MJD": days}
            print(aux_dict["stamp"].shape)
            print(aux_dict["time"]["MJD"].shape)
            pickle.dump(aux_dict,
                        open(field+ "_fake_example.npy", "wb"),
                        protocol=2)
            break
