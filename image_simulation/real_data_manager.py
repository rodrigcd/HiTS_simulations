import numpy as np
import h5py
import glob
import pandas
import matplotlib.pyplot as plt

class RealData(object):

    def __init__(self, **kwargs):
        print("Real data keys: ")
        print("EB, RRLyrae, real_constants, sim_constants")
        self.hdf5_path = kwargs["sim_data_path"]
        self.sequence_length = kwargs["sequence_length"]
        self.tensor_length = kwargs["tensor_length"]
        self.real_data_path = kwargs["real_data_path"]

    def return_real_data(self, class_name, label, n_samples="all", filter_by_cond=True, preprocessing=False):

        def to_counts(mag, zp, texp):
            return texp * np.power(10, (mag - zp) / -2.5)

        def match_good_points(real_days, sim_days, quality, zp):
            matched_quality = []
            matched_zp = []
            matched_index = []
            for day in real_days:
                day_diff = np.abs(sim_days - day)
                matched_quality.append(quality[np.argmin(day_diff)])
                matched_zp.append(zp[np.argmin(day_diff)])
                matched_index.append(np.argmin(day_diff))
            matched_quality = np.array(matched_quality)
            matched_zp = np.array(matched_zp)
            matched_index = np.array(matched_index)
            return matched_quality, matched_zp, matched_index

        image_file = h5py.File(self.hdf5_path, "r")

        path = self.real_data_path + class_name + "/"

        path_list = sorted(glob.glob(path + "*.npy"), key=str.lower)
        numpy_file_names = []
        for p in path_list:
            numpy_file_names.append(p.split("/")[-1].split(".")[0].split("_")[:6])
        with_lc = True
        lc_list = sorted(glob.glob(path + "lightcurves/*.dat"))

        mag_list = []
        mag_errors = []
        if not lc_list:
            magnitudes = pandas.read_csv(path + "lightcurves/NV_15A_list.txt", header=None)
            for p in path_list:
                name_parts = p.split("/")[-1].split("_")
                field = "Field" + name_parts[1]
                key_name = name_parts[0] + "_" + name_parts[1] + "_" + name_parts[2] + "_" \
                           + name_parts[3].zfill(4) + "_" + name_parts[4].zfill(4)
                _, m, error = magnitudes[magnitudes[0] == key_name].values[0]
                days = np.load(p, encoding="latin1").item()["time"]["MJD"]
                mag_list.append(np.ones(shape=days.shape) * m)
                mag_errors.append(np.ones(shape=days.shape) * error)
        else:

            for p, split_name in enumerate(numpy_file_names):
                lc_file = path + "lightcurves/"
                lc_file += split_name[0] + "_" + split_name[1] + "_" + split_name[2] + "_" \
                           + split_name[3].zfill(4) + "_" + split_name[4].zfill(4) + "_" + split_name[5]
                lc_file += ".dat"
                frame = pandas.read_csv(lc_file, sep='\t')
                mag_list.append(frame["MAG_KRON"].values)
                mag_errors.append(frame["MAGERR_KRON"].values)

        network_input_data = {"images": [], "days": [], "lengths": [], "field": [],
                              "labels": [], "input_seq": [], "input_days": [], "stamps": [],
                              "lightcurve": [], "filtered_days": [], "filtered_stamps": [],
                              "match_index": [], "camera": [], "lightcurve_mag": [],
                              "original_lc_mag": [], "original_lc_counts":[]}

        if n_samples != "all":
            path_list = path_list[:n_samples]
        for ind, info in enumerate(path_list):
            field = "Field" + info.split("/")[-1].split("_")[1]
            camera = "CCD" + info.split("/")[-1].split("_")[2][1:]
            try:
                data = np.load(info, encoding="latin1").item()
            except:
                data = np.load(info)

            if filter_by_cond:
                good_quality_points = image_file[field]["obs_cond"]["good_quality_points"]["g"][:]
            else:
                good_quality_points = np.ones(shape=(len(image_file[field]["obs_cond"]["good_quality_points"]["g"][:]))).astype(bool)

            # print(np.array_equal(good_quality_points, match_gp))

            zero_point = image_file[field]["obs_cond"]["zero_point"]["g"][:]
            flux_factor = image_file[field]["obs_cond"]["flux_conversion"]["g"][:]
            sky = image_file[field]["obs_cond"]["sky_brightness"]["g"][:]
            sim_data_length = len(zero_point)
            real_data_length = len(data["time"]["MJD"])

            # print(field, real_data_length)
            index = np.arange(start=0, stop=len(data["time"]["MJD"]), step=1)
            if real_data_length != sim_data_length:
                # print("UUPS, no match between sim and real data length in field "+field)
                good_quality_points, zero_point, index = match_good_points(real_days=data["time"]["MJD"],
                                                                           sim_days=
                                                                           image_file[field]["obs_cond"]["obs_day"][
                                                                               "g"][:],
                                                                           quality=good_quality_points,
                                                                           zp=zero_point)
                flux_factor = flux_factor[index]
                sky = sky[index]
            orig_zero_point = zero_point
            zero_point = zero_point[good_quality_points]
            flux_factor = flux_factor[good_quality_points]
            sky = sky[good_quality_points]
            # print(field)
            # print(info.split("/")[-1])
            # for d, day in enumerate(image_file[field]["obs_cond"]["obs_day"]["g"][:]):

            stamp = data["stamp"][good_quality_points, ...]
            lc = to_counts(mag_list[ind][good_quality_points], zero_point, 86.0)
            if preprocessing:
                stamp = stamp - sky[:, np.newaxis, np.newaxis]
                stamp = np.multiply(stamp, flux_factor[:, np.newaxis, np.newaxis])
                # plt.imshow(stamp[-1, :, :])
                # plt.colorbar()
                # plt.show()
                lc = np.multiply(lc, flux_factor)
            network_input_data["filtered_stamps"].append(stamp)
            network_input_data["filtered_days"].append(
                data["time"]["MJD"][good_quality_points][:self.sequence_length + self.tensor_length])
            network_input_data["days"].append(data["time"]["MJD"])
            network_input_data["stamps"].append(data["stamp"])
            network_input_data["labels"].append(label)
            network_input_data["field"].append(field)
            network_input_data["lightcurve"].append(lc)
            network_input_data["match_index"].append(index)
            network_input_data["camera"].append(camera)
            network_input_data["original_lc_mag"].append(mag_list[ind])
            network_input_data["original_lc_counts"].append(to_counts(mag_list[ind],
                                                                      orig_zero_point,
                                                                      86.0))
            network_input_data["lightcurve_mag"].append(mag_list[ind][good_quality_points])
            # plt.plot(network_input_data["days"][-1], lc, label="Transformed")
            # plt.plot(network_input_data["days"][-1], to_counts(mag_list[ind][good_quality_points], zero_point, 86.0),
            #         label="Original")
            # plt.plot(network_input_data["days"][-1], to_counts(mag_list[ind][good_quality_points], zero_point[0], 86.0),
            #         label="Constant")
            # plt.legend()
            # plt.show()

            input_sequence = []
            input_days = []
            # print(network_input_data["images"][-1].shape)
            seq_len = 0
            for i in range(self.sequence_length):
                seq = np.arange(start=i, stop=i + self.tensor_length, step=1)
                if seq[2] >= len(network_input_data["filtered_days"][-1]):
                    break
                # print(seq)
                input_sequence.append(np.moveaxis(network_input_data["filtered_stamps"][-1][seq, ...], 0, -1))
                # print(input_sequence[-1].shape)
                input_days.append(network_input_data["filtered_days"][-1][seq])
                seq_len += 1
            network_input_data["lengths"].append(seq_len)

            if len(input_sequence) < self.sequence_length:
                fill_length = self.sequence_length - len(input_sequence)
                # print(seq[2])
                # print(fill_length)
                network_input_data["input_seq"].append(np.stack(input_sequence \
                                                                + [np.zeros(
                    shape=input_sequence[-1].shape), ] * fill_length,
                                                                axis=0))
                network_input_data["input_days"].append(np.stack(input_days \
                                                                 + [np.zeros(
                    shape=(self.tensor_length)), ] * fill_length,
                                                                 axis=0))
                # print("not enough length")
                # print(network_input_data["input_seq"][-1].shape)
            else:
                network_input_data["input_seq"].append(np.stack(input_sequence, axis=0))
                network_input_data["input_days"].append(np.stack(input_days, axis=0))
                # print("enough length")
                # print(network_input_data["input_seq"][-1].shape)

        make_numpy_array = ["lengths", "labels", "input_seq", "input_days"]
        for key in make_numpy_array:
            # print("Key " + key)
            network_input_data[key] = np.stack(network_input_data[key], axis=0)
            # print(network_input_data[key].shape)
        network_input_data["n_examples"] = len(network_input_data["labels"])
        return network_input_data

    def return_supernovae(self, n_sigma_detection = 4.5, filter_by_condition=True, return_data=False, preprocessing=False):

        image_file = h5py.File(self.hdf5_path, "r")

        def detection_time(stamp_days, frame, good_quality_points, field_days):
            counts = frame["ADU"].values
            err_counts = frame["e_ADU"].values
            frame_days = frame["MJD"].values

            if len(frame_days) != len(field_days):
                print("different lengths days")
            right_index = []
            for day in frame_days:
                day_diff = np.abs(field_days - day)
                right_index.append(np.argmin(day_diff))
            right_index = np.array(right_index)
            good_quality_points = good_quality_points[right_index]
            counts = counts[good_quality_points]
            err_counts = err_counts[good_quality_points]
            frame_days = frame_days[good_quality_points]

            detection_index = np.where(counts-err_counts*n_sigma_detection>0)[0]
            if len(detection_index)==0:
                return [], [], []
            detection_day_lc = frame_days[detection_index[0]]
            detection_index_stamp = np.argmin(np.abs(stamp_days-detection_day_lc))
            images_after_detection = len(frame_days)-detection_index_stamp
            if images_after_detection > 20:
                images_after_detection = 20
            return detection_index_stamp, images_after_detection, right_index

        real_data_path = "./real_data/SNstamps/npy/brighter/"
        real_lc_path = "./real_data/SNstamps/npy/brighter/"
        real_data_stamps = sorted(glob.glob(real_data_path + "*stamps.npy"), key=str.lower)
        real_data_mjds = sorted(glob.glob(real_data_path + "*MJDs.npy"), key=str.lower)
        lc_path_list = sorted(glob.glob(real_lc_path + "*dat"), key=str.lower)
        plot_distr = False

        def plot_magnitude_distribution(list_lc):
            mag_list = []
            for frame in list_lc:
                # print(type(frame))
                mag_list.append(np.amin(frame["mag"].values))
            bins = np.arange(start=14, stop=26, step=0.5)
            h, _ = np.histogram(mag_list, bins=bins, density=True)
            plt.plot(bins[1:], h)
            plt.show()

        image_file = h5py.File(self.hdf5_path, "r")

        stamp_names = []
        lc_names = []
        matched_lc = []
        matched_stamps = []
        matched_mjds = []
        n_matched = 0
        for stamp_name in real_data_stamps:
            name = stamp_name.split("/")[-1]
            id_name = name.split("_")[0]
            stamp_names.append(id_name)
        stamp_names = np.array(stamp_names)

        for k, lc_name in enumerate(lc_path_list):
            name = lc_name.split("/")[-1]
            id_name = name.split(".")[0]
            lc_names.append(id_name)
            if id_name in stamp_names:
                matched_lc.append(lc_name)
                index = np.where(stamp_names == id_name)[0]
                # print(index)
                matched_stamps.append(real_data_stamps[index[0]])
                matched_mjds.append(real_data_mjds[index[0]])
                n_matched += 1

        network_input_data = {"images": [], "days": [], "lengths": [], "field":[],
                              "labels": [], "input_seq": [], "input_days": [], "stamps":[],
                              "detection_index": [], "lightcurve":[], "filtered_stamps":[],
                              "filtered_days":[]}
        #print(matched_lc)
        frame_list = []
        max_length = 0
        field_data = pandas.read_csv("./real_data/SNstamps/SNHiTS.txt", delimiter=r"\s+")
        #if filter_by_condition:
        #    good_quality_points = image_file[]

        for i, stamp_path in enumerate(matched_stamps):
            field = field_data[field_data["SNname"]==matched_lc[i].split("/")[-1].split(".")[0]]["field"].values[0]
            field = "Field"+str(field).zfill(2)
            print(field)
            stamp = np.load(stamp_path).astype(np.float32)
            mjds = np.load(matched_mjds[i]).astype(np.float32)
            if filter_by_condition:
                good_quality_points = image_file[field]["obs_cond"]["good_quality_points"]["g"][:]
            else:
                good_quality_points = np.ones(shape=mjds.shape).astype(bool)
            lc_frame = pandas.read_csv(matched_lc[i], sep=' ', comment="#")
            lc_frame = lc_frame[lc_frame["band"] == "g"]
            frame_list.append(lc_frame)
            #print([len(lc_frame), len(mjds), len(field_days)])
            field_days = image_file[field]["obs_cond"]["obs_day"]["g"][:]
            shorter_len = np.amin([len(lc_frame), len(mjds), len(field_days)])
            field_days = field_days[:shorter_len]
            zp = image_file[field]["obs_cond"]["zero_point"]["g"][:shorter_len]
            sky = image_file[field]["obs_cond"]["sky_brightness"]["g"][:shorter_len]
            flux_factor = image_file[field]["obs_cond"]["flux_conversion"]["g"][:shorter_len]
            det_index, length, right_index = detection_time(mjds, lc_frame,
                                                            good_quality_points, field_days)
            if len(right_index) == 0:
                print("Not detected")
                continue
            print(len(lc_frame["MJD"].values), len(mjds), len(field_days))
            field_days = field_days[right_index]
            good_quality_points = good_quality_points[right_index]
            zp = zp[right_index]
            sky = sky[right_index]
            flux_factor = flux_factor[right_index]
            lc = lc_frame["ADU"].values
            stamp = stamp[right_index, ...]

            if preprocessing:
                stamp = stamp - sky[:, np.newaxis, np.newaxis]
                stamp = np.multiply(stamp, flux_factor[:, np.newaxis, np.newaxis])
                #plt.imshow(stamp[-1, :, :])
                #plt.colorbar()
                #plt.show()
                lc = np.multiply(lc, flux_factor)

            network_input_data["filtered_stamps"].append(stamp[good_quality_points, ...])
            network_input_data["filtered_days"].append(field_days[good_quality_points, ...])
            network_input_data["lengths"].append(length)
            network_input_data["labels"].append(0)
            network_input_data["detection_index"].append(det_index)
            network_input_data["field"].append(field)
            network_input_data["stamps"].append(stamp)
            network_input_data["days"].append(mjds)
            network_input_data["lightcurve"].append(lc)

            input_sequence = []
            input_days = []
            if len(network_input_data["filtered_days"][-1]) > max_length:
                max_length = len(mjds)

            for j in range(np.amin([length, self.sequence_length])):
                seq = np.arange(start=network_input_data["detection_index"][-1]-3+j,
                                stop=network_input_data["detection_index"][-1]-3+j + self.tensor_length,
                                step=1)
                input_sequence.append(np.moveaxis(network_input_data["filtered_stamps"][-1][seq, ...], 0, -1))
                input_days.append(network_input_data["filtered_days"][-1][seq])

            for j in range(self.sequence_length-len(input_sequence)):
                input_sequence.append(np.zeros(shape=(21, 21, self.tensor_length)))
                input_days.append(np.zeros(shape=(3,)))

            network_input_data["input_seq"].append(np.stack(input_sequence, axis=0))
            network_input_data["input_days"].append(np.stack(input_days, axis=0))


            #print(network_input_data["input_seq"][-1].shape)
            #print(network_input_data["input_seq"][-1].shape)
        make_numpy_array = ["lengths", "labels", "input_seq", "input_days"]
        for key in make_numpy_array:
            print("Key "+key)
            if key=="input_seq":
                for d in network_input_data[key]:
                    print(d.shape)
            network_input_data[key] = np.stack(network_input_data[key], axis=0)
            print(network_input_data[key].shape)
        network_input_data["n_examples"] = len(network_input_data["labels"])

        print("n supernovae: " + str(len(network_input_data["images"])))
        print("max length: " + str(max_length))

        return network_input_data

if __name__ == "__main__":
    sim_data_path = "/home/rodrigo/supernovae_detection/simulated_data/image_sequences/small_may30_erf_distr50.hdf5"
    real_data = RealData(sim_data_path=sim_data_path,
                         sequence_length=20,
                         tensor_length=3,
                         real_data_path="../real_obs/real_stamps/")
    rr_lyrae_data = real_data.return_real_data("real_constants", label=1, filter_by_cond=False)